from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from django.core.files.base import ContentFile
from django_ai_sdk.adapters.base import _SENTINEL, Stream
from django_ai_sdk.agents.base import Agent
from django_ai_sdk.artifacts import FileArtifact, ToolArtifact
from django_ai_sdk.common import StreamWriter
from django_ai_sdk.memories.tools import ASK_IMAGE_ARTIFACT, ASK_IMAGE_TOOL
from haystack.dataclasses import StreamingChunk, ToolCall, ToolCallDelta, ToolCallResult


def _result_chunk(tool_name: str, arguments: dict, result: str, error: bool = False):
    call = ToolCall(id=f"call-{tool_name}", tool_name=tool_name, arguments=arguments)
    return StreamingChunk(
        content="",
        index=0,
        tool_call_result=ToolCallResult(result=result, origin=call, error=error),
    )


async def _events(stream: Stream, *chunks: StreamingChunk, writer: StreamWriter | None = None):
    queue: asyncio.Queue = asyncio.Queue()
    for chunk in chunks:
        await queue.put(chunk)
    await queue.put(_SENTINEL)
    return [e async for e in stream.get_events(queue, writer)]


def _stream(tool_artifacts: dict, thread_id: str | None = "t") -> Stream:
    stream = Stream.__new__(Stream)
    stream.citation_registry = None
    stream._persisted_tool_ids, stream._persisted_tool_output_ids = set(), set()
    stream._handoff_tools = {}
    stream.tool_artifacts, stream.thread_id, stream.user = tool_artifacts, thread_id, None
    return stream


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

    async def test_ask_image_result_is_followed_by_a_file_artifact(self):
        from django_ai_sdk.artifacts.models import Artifact

        thread_id, doc_id = await self._thread_image()
        stream = _stream({ASK_IMAGE_TOOL: ASK_IMAGE_ARTIFACT}, thread_id)
        writer = StreamWriter(message_id="m1")

        arguments = {"image": "cat", "question": "?"}
        delta = ToolCallDelta(
            index=0,
            id=f"call-{ASK_IMAGE_TOOL}",
            tool_name=ASK_IMAGE_TOOL,
            arguments=json.dumps(arguments),
        )
        events = await _events(
            stream,
            StreamingChunk(content="", index=0, tool_calls=[delta]),  # the model's call
            _result_chunk(ASK_IMAGE_TOOL, arguments, "grey"),
            writer=writer,
        )

        # The tool call and its output first, then the artifact as its own tool call.
        assert [e.event_type for e in events] == [
            "tool_call_start",
            "tool_input_complete",
            "tool_output",
            "tool_call_start",
            "tool_input_complete",
            "tool_output",
        ]
        payload = events[-1].tool_output
        assert payload["files"][0]["documentId"] == doc_id
        assert payload["files"][0]["filename"] == "cat.png"
        artifact = await Artifact.objects.aget(id=payload["artifact_id"])
        assert artifact.schema_name == "FileArtifact"
        # Stored with the message, so a reloaded thread shows it too.
        names = [c["name"] for c in writer.message.tool_calls]
        assert names == [ASK_IMAGE_TOOL, "artifact_file_artifact"]

    async def test_builder_gets_the_tool_result_as_json_data(self):
        thread_id, _ = await self._thread_image()
        build = AsyncMock(return_value=None)
        stream = _stream({"lookup": ToolArtifact(FileArtifact, build)}, thread_id)

        await _events(stream, _result_chunk("lookup", {"q": "x"}, json.dumps([{"entry_id": "e1"}])))

        build.assert_awaited_once_with({"q": "x"}, [{"entry_id": "e1"}], thread_id)

    @pytest.mark.parametrize(
        "case",
        ["builder returns None", "rejected data", "unmapped tool", "tool failed", "no thread"],
    )
    async def test_no_artifact(self, case):
        from django_ai_sdk.artifacts.models import Artifact

        thread_id, _ = await self._thread_image()
        data = None if case == "builder returns None" else {"files": []}  # [] fails FileData
        spec = ToolArtifact(FileArtifact, lambda args, result, tid: data)
        stream = _stream({"lookup": spec}, None if case == "no thread" else thread_id)
        name = "other" if case == "unmapped tool" else "lookup"

        events = await _events(
            stream, _result_chunk(name, {}, "found", error=case == "tool failed")
        )

        assert [e.event_type for e in events] == ["tool_output"]
        assert not await Artifact.objects.filter(thread_id=thread_id).aexists()


@pytest.mark.asyncio
async def test_stream_gets_the_agents_tool_artifacts():
    from django_ai_sdk.agents import ToolAgent, ToolAgentConfig
    from django_ai_sdk.permissions import AllowAll
    from haystack.components.generators.chat import MockChatGenerator

    spec = ToolArtifact(FileArtifact, AsyncMock(return_value=None))

    class FileAgent(Agent):
        permissions = [AllowAll]
        tool_artifacts = {"lookup": spec}

        async def get_pipeline_adapter(self, thread_id=None, user=None):
            config = ToolAgentConfig(model="m", system_prompt="s", tools=[])
            generator = MockChatGenerator(responses=["hi"])
            pipeline = ToolAgent(config, generator=generator).pipeline()
            return Stream(pipeline=pipeline, generator=generator)

    with patch("django_ai_sdk.agents.base.stream_response", new=AsyncMock()) as respond:
        await FileAgent().as_view([], thread_id="t")
        adapter = await respond.await_args.args[0]()

    assert (adapter.tool_artifacts, adapter.thread_id) == ({"lookup": spec}, "t")
