from __future__ import annotations

import base64
from typing import ClassVar
from types import SimpleNamespace

import pytest
from django.core.files.base import ContentFile
from haystack.components.agents.state.state import State
from haystack.dataclasses import ChatMessage as HaystackChatMessage

from django_ai_sdk.adapters.base import get_user_message
from django_ai_sdk.agents.base import Agent
from django_ai_sdk.agents.tool_agent import RequireToolsHook
from django_ai_sdk.common import Attachment, ChatMessage
from django_ai_sdk.protocols.vercel import VercelProtocolHandler
from django_ai_sdk.views.schemas import Message

PNG = b"\x89PNG\r\n\x1a\nfake"


def _file_part(document_id: str, media_type: str = "image/png") -> dict:
    return {
        "type": "file",
        "mediaType": media_type,
        "filename": "evil.png",
        "url": "http://169.254.169.254/latest",
        "providerMetadata": {"sdk": {"documentId": document_id}},
    }


class TestProtocol:
    def test_file_part_becomes_attachment_and_round_trips(self):
        handler = VercelProtocolHandler()
        msg = Message.model_validate({"role": "user", "parts": [_file_part("doc-1")]})

        [chat] = handler.to_chat_messages([msg])

        # image-only message is kept, url from the client is not carried
        assert chat.content == ""
        assert chat.attachments[0].document_id == "doc-1"
        assert chat.attachments[0].url == ""

        chat.attachments[0].url = "/media/a.png"
        [out] = handler.from_chat_messages([chat])
        assert out["parts"][0] == {
            "type": "file",
            "mediaType": "image/png",
            "filename": "evil.png",
            "url": "/media/a.png",
            "providerMetadata": {"sdk": {"documentId": "doc-1"}},
        }

    def test_request_only_fields_are_not_persisted(self):
        attachment = Attachment(document_id="d", url="u", context="c", data="x")
        dumped = ChatMessage(role="user", attachments=[attachment]).model_dump()
        assert dumped["attachments"] == [
            {"document_id": "d", "media_type": "", "filename": "", "memory_id": ""}
        ]


class TestUserMessage:
    def test_hint_only_without_data(self):
        msg = ChatMessage(
            role="user", content="hi", attachments=[Attachment(document_id="d", context="[file]")]
        )
        result = get_user_message(msg)
        assert result.text == "hi\n\n[file]"
        assert not result.images

    def test_inlined_image_is_sent_as_image_content(self):
        data = base64.b64encode(PNG).decode()
        msg = ChatMessage(
            role="user",
            content="what is this?",
            attachments=[Attachment(document_id="d", media_type="image/png", data=data)],
        )
        result = get_user_message(msg)
        assert result.images[0].base64_image == data


@pytest.mark.django_db
@pytest.mark.asyncio
class TestResolveAttachments:
    @pytest.fixture(autouse=True)
    def tmp_media(self, tmp_path, settings):
        settings.MEDIA_ROOT = str(tmp_path / "media")

    @staticmethod
    async def _user():
        from tests.factories.db import UserFactory

        return await UserFactory.acreate()

    @staticmethod
    def _agent(**formatters):
        from django_ai_sdk.permissions import AllowAll

        # The thread's own permissions decide in these tests.
        return SimpleNamespace(permissions=[AllowAll], is_runtime=False, **formatters)

    async def _thread_with_image(self, user=None):
        from django_ai_sdk.conversation.models import Thread
        from django_ai_sdk.memories.models import EntryDocument, Memory

        memory = await Memory.objects.acreate(name="files", description="Thread file uploads")
        thread = await Thread.objects.acreate(file_memory=memory, user=user)
        doc = EntryDocument(
            memory=memory, file_name="cat.png", file_size=len(PNG), content_type="image/png"
        )
        doc.file.save("cat.png", ContentFile(PNG), save=False)
        await doc.asave()
        return thread, doc

    async def test_foreign_ids_dropped_and_last_image_inlined(self):
        from django_ai_sdk.memories.services import MemoryService

        owner = await self._user()
        thread, doc = await self._thread_with_image(owner)
        other_thread, other_doc = await self._thread_with_image(owner)
        agent = self._agent(
            format_attachment=lambda d: f"[{d.file_name}]",
            format_image_attachment=lambda d, inline: f"[{d.file_name} inline={inline}]",
        )
        messages = [
            ChatMessage(role="user", attachments=[Attachment(document_id=str(doc.id))]),
            ChatMessage(role="assistant", content="a cat"),
            ChatMessage(
                role="user",
                attachments=[
                    Attachment(document_id=str(doc.id), filename="evil.exe"),
                    Attachment(document_id=str(other_doc.id)),
                    Attachment(document_id="not-a-uuid"),
                ],
            ),
        ]

        await MemoryService.resolve_attachments(
            str(thread.id), messages, user=owner, agent=agent, inline_images=True
        )

        first, _, last = messages
        assert [a.document_id for a in last.attachments] == [str(doc.id)]
        assert last.attachments[0].filename == "cat.png"
        assert last.attachments[0].context == "[cat.png inline=True]"
        assert base64.b64decode(last.attachments[0].data) == PNG
        # older turns get the hint only
        assert first.attachments[0].data == ""
        assert first.attachments[0].context == "[cat.png inline=False]"

    async def test_oversized_image_not_inlined(self, settings):
        from django_ai_sdk.memories.services import MemoryService

        owner = await self._user()
        settings.AI_SDK_MAX_INLINE_IMAGE_BYTES = 1
        thread, doc = await self._thread_with_image(owner)
        agent = self._agent(
            format_attachment=lambda d: "", format_image_attachment=lambda d, inline: ""
        )
        messages = [ChatMessage(role="user", attachments=[Attachment(document_id=str(doc.id))])]

        await MemoryService.resolve_attachments(
            str(thread.id), messages, user=owner, agent=agent, inline_images=True
        )
        assert messages[0].attachments[0].data == ""

    async def test_someone_elses_thread_files_are_not_read(self):
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.permissions import PermissionDenied

        owner = await self._user()
        stranger = await self._user()
        thread, doc = await self._thread_with_image(owner)
        agent = self._agent(
            format_attachment=lambda d: "", format_image_attachment=lambda d, inline: ""
        )
        messages = [ChatMessage(role="user", attachments=[Attachment(document_id=str(doc.id))])]

        with pytest.raises(PermissionDenied):
            await MemoryService.resolve_attachments(
                str(thread.id), messages, user=stranger, agent=agent, inline_images=True
            )
        # Nothing was read or rewritten.
        assert messages[0].attachments[0].data == ""
        assert messages[0].attachments[0].filename != "cat.png"


class TestRequireToolsHook:
    def _state(self, counts: dict, hook_context: dict | None = None) -> State:
        from haystack.components.agents.agent import _INTERNAL_STATE_KEYS, _RUN_METADATA_STATE_KEYS

        schema = {
            "messages": {"type": list[HaystackChatMessage]},
            **_RUN_METADATA_STATE_KEYS,
            **_INTERNAL_STATE_KEYS,
        }
        state = State(schema=schema)
        state.set("tools", [SimpleNamespace(name="get_today"), SimpleNamespace(name="read_file")])
        state.set("tool_call_counts", counts)
        state.set("hook_context", hook_context or {})
        state.set("continue_run", False)
        return state

    def test_keeps_running_until_required_tool_called(self):
        state = self._state({})
        RequireToolsHook(["get_today"]).run(state)
        assert state.get("continue_run") is True

        state = self._state({"get_today": 1})
        RequireToolsHook(["get_today"]).run(state)
        assert state.get("continue_run") is False

    def test_run_context_and_missing_tools(self):
        state = self._state({}, {"required_tools": ["read_file"]})
        RequireToolsHook().run(state)
        assert state.get("continue_run") is True

        state = self._state({})
        RequireToolsHook(["not_in_toolset"]).run(state)
        assert state.get("continue_run") is False

    def test_nudges_are_capped(self):
        hook = RequireToolsHook(["get_today"], max_nudges=1)
        state = self._state({})
        hook.run(state)
        state.set("continue_run", False)
        hook.run(state)
        assert state.get("continue_run") is False


class TestVisionFallback:
    def _agent(self, vision=False, model="text-model"):
        return SimpleNamespace(vision=vision, get_model=lambda: model)

    def test_has_vision(self, settings):
        from django_ai_sdk.agents.base import Agent

        settings.AI_SDK_VISION_MODEL = "vision-model"
        assert Agent.has_vision(self._agent(vision=True))
        assert Agent.has_vision(self._agent(model="vision-model"))
        assert not Agent.has_vision(self._agent())

        settings.AI_SDK_VISION_MODEL = None
        assert not Agent.has_vision(self._agent())

    def test_image_hint_falls_back_to_caption(self, settings):
        from django_ai_sdk.agents.base import Agent

        doc = SimpleNamespace(
            id="d1", file_name="cat.png", entry=SimpleNamespace(content="A grey cat.")
        )
        assert "included in this message" in Agent.format_image_attachment(None, doc, inline=True)

        settings.AI_SDK_VISION_MODEL = "vision-model"
        hint = Agent.format_image_attachment(None, doc, inline=False)
        assert "A grey cat." in hint
        assert "ask_image" in hint

        settings.AI_SDK_VISION_MODEL = None
        hint = Agent.format_image_attachment(None, doc, inline=False)
        assert "A grey cat." in hint
        assert "ask_image" not in hint

    def test_a_long_caption_is_capped_by_the_setting(self, settings):
        from django_ai_sdk.agents.base import Agent

        doc = SimpleNamespace(id="d1", file_name="cat.png", entry=SimpleNamespace(content="x" * 50))
        settings.AI_SDK_VISION_MODEL = None

        settings.AI_SDK_IMAGE_CAPTION_LIMIT = 10
        assert "x" * 10 + "…" in Agent.format_image_attachment(None, doc, inline=False)
        assert "x" * 11 not in Agent.format_image_attachment(None, doc, inline=False)

        settings.AI_SDK_IMAGE_CAPTION_LIMIT = 0  # off
        assert "x" * 50 in Agent.format_image_attachment(None, doc, inline=False)

        settings.AI_SDK_IMAGE_CAPTION_LIMIT = None  # off too, like AI_SDK_MAX_INLINE_IMAGE_BYTES
        assert "x" * 50 in Agent.format_image_attachment(None, doc, inline=False)

    def test_images_do_not_require_attachment_tools(self):
        from django_ai_sdk.agents.base import Agent

        agent = SimpleNamespace(attachment_tools=["get_memory_file"])
        image = Attachment(document_id="a", media_type="image/png")
        pdf = Attachment(document_id="b", media_type="application/pdf")
        only_image = [ChatMessage(role="user", attachments=[image])]
        with_pdf = [ChatMessage(role="user", attachments=[image, pdf])]

        assert Agent.get_run_required_tools(agent, only_image) == []
        assert Agent.get_run_required_tools(agent, with_pdf) == ["get_memory_file"]


@pytest.mark.django_db
@pytest.mark.asyncio
class TestAskImageTool:
    @pytest.fixture(autouse=True)
    def tmp_media(self, tmp_path, settings):
        settings.MEDIA_ROOT = str(tmp_path / "media")

    async def _thread(self, name="cat.png", content_type="image/png"):
        from django_ai_sdk.conversation.models import Thread
        from django_ai_sdk.memories.models import EntryDocument, Memory

        memory = await Memory.objects.acreate(name=f"files-{name}")
        thread = await Thread.objects.acreate(file_memory=memory)
        doc = EntryDocument(memory=memory, file_name=name, content_type=content_type)
        doc.file.save(name, ContentFile(PNG), save=False)
        await doc.asave()
        return thread, doc

    async def test_answers_only_for_this_threads_images(self):
        from unittest.mock import AsyncMock, patch

        from django_ai_sdk.memories.tools import ask_image_tool, get_thread_image

        thread, doc = await self._thread()
        _, foreign = await self._thread("dog.png")
        _, pdf = await self._thread("notes.pdf", "application/pdf")

        assert (await get_thread_image(str(thread.id), str(doc.id))).id == doc.id
        assert (await get_thread_image(str(thread.id), "CAT")).id == doc.id
        assert await get_thread_image(str(thread.id), str(foreign.id)) is None
        assert await get_thread_image(str(thread.id), str(pdf.id)) is None

        tool = ask_image_tool(str(thread.id))
        with patch(
            "django_ai_sdk.memories.tools.describe_image", new=AsyncMock(return_value="grey")
        ) as describe:
            assert await tool.async_function(image=str(doc.id), question="colour?") == "grey"
            assert describe.await_args.args == (PNG, "image/png", "colour?")
            missing = await tool.async_function(image=str(foreign.id), question="colour?")
        assert "No image" in missing


class RecordingVisionAgent(Agent):
    """A project's own vision agent, as AI_SDK_VISION_AGENT would name it."""

    hidden = True
    model = "vision-test"
    calls: ClassVar[list] = []

    async def get_pipeline_adapter(self, thread_id=None, user=None):
        raise NotImplementedError

    async def run(self, messages, **kwargs):
        type(self).calls.append((messages, kwargs))
        return "a grey cat"


VISION_AGENT = f"{__name__}.RecordingVisionAgent"


class TestVisionAgent:
    async def test_the_vision_agent_answers_with_the_image_attached(self, settings):
        from django_ai_sdk.files.processors import describe_image

        settings.AI_SDK_VISION_AGENT = VISION_AGENT
        RecordingVisionAgent.calls.clear()

        assert await describe_image(PNG, "image/png", "What is it?") == "a grey cat"

        ((messages, kwargs),) = RecordingVisionAgent.calls
        (message,) = messages
        assert message.content == "What is it?"
        (image,) = message.attachments
        assert (image.media_type, base64.b64decode(image.data)) == ("image/png", PNG)
        assert kwargs["response_format"] is None

    async def test_without_an_agent_the_vision_model_is_called(self, settings):
        from unittest.mock import AsyncMock, MagicMock, patch

        from django_ai_sdk.files.processors import describe_image

        settings.AI_SDK_VISION_AGENT = None
        settings.AI_SDK_VISION_MODEL = "vision-model"
        generator = MagicMock()
        generator.run_async = AsyncMock(return_value={"replies": [MagicMock(text="a dog")]})
        with patch("django_ai_sdk.generators.openai_chat", return_value=generator) as chat:
            assert await describe_image(PNG, "image/png", "What is it?") == "a dog"
        chat.assert_called_once_with(model="vision-model")

    def test_vision_support_needs_an_agent_or_a_model(self, settings):
        from django_ai_sdk.files.processors import has_vision_support

        settings.AI_SDK_VISION_AGENT, settings.AI_SDK_VISION_MODEL = None, None
        assert not has_vision_support()
        settings.AI_SDK_VISION_AGENT = VISION_AGENT
        assert has_vision_support()
        settings.AI_SDK_VISION_AGENT, settings.AI_SDK_VISION_MODEL = None, "vision-model"
        assert has_vision_support()

    def test_the_setting_must_name_an_agent(self, settings):
        from django.core.exceptions import ImproperlyConfigured

        from django_ai_sdk.files.processors import get_vision_agent

        settings.AI_SDK_VISION_AGENT = "django_ai_sdk.common.ChatMessage"
        with pytest.raises(ImproperlyConfigured, match="Agent subclass"):
            get_vision_agent()
