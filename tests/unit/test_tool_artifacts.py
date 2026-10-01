from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from django.core.files.base import ContentFile
from django_ai_sdk.adapters.base import _SENTINEL, Stream
from django_ai_sdk.agents.base import Agent
from django_ai_sdk.artifacts import FileArtifact, ToolArtifact, with_tool_artifact
from django_ai_sdk.common import StreamWriter
from django_ai_sdk.memories.tools import ASK_IMAGE_ARTIFACT, ask_image_tool
from haystack import Pipeline
from haystack.tools import Tool


def _plain_tool(name: str = "lookup") -> Tool:
    return Tool(
        name=name,
        description="",
        parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        function=lambda q: f"found {q}",
    )


@pytest.mark.django_db
@pytest.mark.asyncio
class TestToolArtifacts:
    @pytest.fixture(autouse=True)
    def tmp_media(self, tmp_path, settings):
        settings.MEDIA_ROOT = str(tmp_path / "media")

    async def _thread_image(self):
        from django_ai_sdk.conversation.models import Thread
        from django_ai_sdk.memories.models import EntryDocument, Memory

        memory = await Memory.objects.acreate(name="files")
        thread = await Thread.objects.acreate(file_memory=memory)
        doc = EntryDocument(memory=memory, file_name="cat.png", content_type="image/png")
        doc.file.save("cat.png", ContentFile(b"png"), save=False)
        await doc.asave()
        return str(thread.id), str(doc.id)

    async def test_ask_image_streams_a_file_artifact_after_its_result(self):
        from django_ai_sdk.artifacts.models import Artifact

        thread_id, doc_id = await self._thread_image()
        tool = with_tool_artifact(ask_image_tool(thread_id), ASK_IMAGE_ARTIFACT, thread_id=thread_id)
        queue: asyncio.Queue = asyncio.Queue()

        with patch("django_ai_sdk.memories.tools.describe_image", new=AsyncMock(return_value="grey")):
            result = await tool.invoke_async(
                image="cat", question="colour?", streaming_callback=queue.put
            )

        # The model only sees the tool's own answer.
        assert result == "grey"
        artifact = await Artifact.objects.aget(thread_id=thread_id)
        assert artifact.schema_name == "FileArtifact"
        assert artifact.data["files"][0]["documentId"] == doc_id

        # The chunks become an artifact tool call + output, stored on the message.
        await queue.put(_SENTINEL)
        writer = StreamWriter(message_id="m1")
        events = [e async for e in Stream(pipeline=Pipeline(), generator=None).get_events(queue, writer)]
        assert [e.event_type for e in events] == [
            "tool_call_start",
            "tool_input_complete",
            "tool_output",
        ]
        [call] = writer.message.tool_calls
        assert call["name"] == "artifact_file_artifact"

    async def test_no_artifact_without_stream_or_match(self):
        from django_ai_sdk.artifacts.models import Artifact

        thread_id, _ = await self._thread_image()
        tool = with_tool_artifact(ask_image_tool(thread_id), ASK_IMAGE_ARTIFACT, thread_id=thread_id)
        queue: asyncio.Queue = asyncio.Queue()

        with patch("django_ai_sdk.memories.tools.describe_image", new=AsyncMock(return_value="grey")):
            # Agent.run(): no streaming callback.
            assert await tool.invoke_async(image="cat", question="?") == "grey"
            # Unknown image: the builder returns None.
            missing = await tool.invoke_async(image="dog", question="?", streaming_callback=queue.put)

        assert "No image" in missing
        assert queue.empty()
        assert not await Artifact.objects.filter(thread_id=thread_id).aexists()

    async def test_rejected_data_leaves_the_result_alone(self):
        thread_id, _ = await self._thread_image()
        spec = ToolArtifact(FileArtifact, lambda args, result, tid: {"files": [{"documentId": "nope"}]})
        tool = with_tool_artifact(_plain_tool(), spec, thread_id=thread_id)
        queue: asyncio.Queue = asyncio.Queue()

        assert await tool.invoke_async(q="x", streaming_callback=queue.put) == "found x"
        assert queue.empty()



@pytest.mark.asyncio
async def test_stream_wraps_every_mapped_tool_of_the_run_once():
    from django_ai_sdk.agents import ToolAgent, ToolAgentConfig
    from django_ai_sdk.permissions import AllowAll
    from haystack.components.generators.chat import MockChatGenerator

    spec = ToolArtifact(FileArtifact, AsyncMock(return_value=None))

    class FileAgent(Agent):
        permissions = [AllowAll]
        tools = [lambda **kwargs: [_plain_tool(), _plain_tool("other")]]
        tool_artifacts = {"lookup": spec, "rag": spec}

        async def get_pipeline_adapter(self, thread_id=None, user=None):
            tools = await self.get_tools(thread_id=thread_id or "", user=user)
            tools.append(_plain_tool("rag"))  # added after get_tools, like RAG tools
            config = ToolAgentConfig(model="m", system_prompt="s", tools=tools)
            generator = MockChatGenerator(responses=["hi"])
            return Stream(pipeline=ToolAgent(config, generator=generator).pipeline(), generator=generator)

    with patch("django_ai_sdk.agents.base.stream_response", new=AsyncMock()) as respond:
        await FileAgent().as_view([], thread_id="t")
        adapter = await respond.await_args.args[0]()

    by_name = {t.name: t for t in adapter.tools}
    assert by_name["lookup"].function is None
    assert by_name["rag"].function is None
    assert by_name["other"].function is not None
    # Only this run gets the wrapped tools; the agent component keeps its own.
    assert all(t.function is not None for t in adapter.agent_component.tools)

    queue: asyncio.Queue = asyncio.Queue()
    await by_name["lookup"].invoke_async(q="x", streaming_callback=queue.put)
    await by_name["rag"].invoke_async(q="x", streaming_callback=queue.put)
    assert spec.build.await_count == 2
