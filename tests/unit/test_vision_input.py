"""Vision input: image round-trip through the Vercel protocol and Haystack adapter."""

from __future__ import annotations

from django_ai_sdk.adapters.base import build_user_message
from django_ai_sdk.common import ChatMessage, ImageAttachment
from django_ai_sdk.protocols.vercel import VercelProtocolHandler, _parse_image_data_url
from django_ai_sdk.views.schemas import Message, MessagePart

JPEG_URL = "data:image/jpeg;base64,QUJD"  # "ABC"


class TestParseImageDataUrl:
    def test_valid_image_data_url(self):
        assert _parse_image_data_url(JPEG_URL) == ("image/jpeg", "QUJD")

    def test_remote_url_ignored(self):
        assert _parse_image_data_url("https://example.com/a.png") is None

    def test_non_image_data_url_ignored(self):
        assert _parse_image_data_url("data:text/plain;base64,QUJD") is None

    def test_non_base64_ignored(self):
        assert _parse_image_data_url("data:image/png,rawbytes") is None

    def test_none(self):
        assert _parse_image_data_url(None) is None

    def test_extra_params_before_base64(self):
        # base64 need not be the only/last param.
        assert _parse_image_data_url("data:image/jpeg;charset=utf-8;base64,QUJD") == (
            "image/jpeg",
            "QUJD",
        )


class TestToChatMessages:
    def setup_method(self):
        self.handler = VercelProtocolHandler()

    def test_text_plus_image(self):
        msg = Message(
            role="user",
            parts=[
                MessagePart(type="text", text="what is this?"),
                MessagePart(type="file", media_type="image/jpeg", url=JPEG_URL),
            ],
        )
        [cm] = self.handler.to_chat_messages([msg])
        assert cm.content == "what is this?"
        assert cm.images == [ImageAttachment(media_type="image/jpeg", data="QUJD")]

    def test_image_only_message_kept(self):
        msg = Message(
            role="user",
            parts=[MessagePart(type="file", media_type="image/png", url="data:image/png;base64,ZZ")],
        )
        [cm] = self.handler.to_chat_messages([msg])
        assert cm.content == ""
        assert len(cm.images) == 1

    def test_media_type_derived_from_url_when_absent(self):
        msg = Message(role="user", parts=[MessagePart(type="file", url=JPEG_URL)])
        [cm] = self.handler.to_chat_messages([msg])
        assert cm.images[0].media_type == "image/jpeg"
        assert cm.images[0].data == "QUJD"

    def test_remote_url_dropped(self):
        # Remote (non-data:) URLs are ignored — the SDK never fetches them
        # server-side (SSRF), so no image is collected.
        msg = Message(
            role="user",
            parts=[
                MessagePart(type="text", text="hi"),
                MessagePart(type="file", media_type="image/png", url="https://cdn.example/x.png"),
            ],
        )
        [cm] = self.handler.to_chat_messages([msg])
        assert cm.images == []

    def test_non_image_file_part_dropped(self):
        msg = Message(
            role="user",
            parts=[
                MessagePart(type="text", text="hi"),
                MessagePart(type="file", media_type="application/pdf", url="data:application/pdf;base64,ZZ"),
            ],
        )
        [cm] = self.handler.to_chat_messages([msg])
        assert cm.content == "hi"
        assert cm.images == []

    def test_image_count_cap(self, settings):
        settings.AI_SDK_MAX_IMAGES_PER_MESSAGE = 1
        msg = Message(
            role="user",
            parts=[
                MessagePart(type="file", url="data:image/png;base64,AA"),
                MessagePart(type="file", url="data:image/png;base64,BB"),
            ],
        )
        [cm] = self.handler.to_chat_messages([msg])
        assert len(cm.images) == 1

    def test_image_byte_cap(self, settings):
        settings.AI_SDK_MAX_IMAGE_BYTES = 1  # ~0 bytes allowed
        msg = Message(
            role="user",
            parts=[
                MessagePart(type="text", text="hi"),
                MessagePart(type="file", url=JPEG_URL),
            ],
        )
        [cm] = self.handler.to_chat_messages([msg])
        assert cm.images == []
        assert cm.content == "hi"

    def test_caps_disabled_with_none(self, settings):
        settings.AI_SDK_MAX_IMAGE_BYTES = None
        settings.AI_SDK_MAX_IMAGES_PER_MESSAGE = None
        msg = Message(
            role="user",
            parts=[MessagePart(type="file", url=JPEG_URL)],
        )
        [cm] = self.handler.to_chat_messages([msg])
        assert len(cm.images) == 1

    def test_camelcase_media_type_alias(self):
        # Vercel AI SDK sends `mediaType`; the schema alias must accept it.
        part = MessagePart.model_validate({"type": "file", "mediaType": "image/webp", "url": JPEG_URL})
        assert part.media_type == "image/webp"


class TestFromChatMessages:
    def test_reemits_inline_image_file_part(self):
        cm = ChatMessage(
            role="user",
            content="hi",
            id="00000000-0000-0000-0000-000000000001",
            images=[ImageAttachment(media_type="image/jpeg", data="QUJD")],
        )
        [out] = VercelProtocolHandler().from_chat_messages([cm])
        assert {"type": "file", "mediaType": "image/jpeg", "url": JPEG_URL} in out["parts"]


class TestBuildUserMessage:
    def test_multimodal_message_has_image_content(self):
        cm = ChatMessage(
            role="user",
            content="describe",
            images=[ImageAttachment(media_type="image/jpeg", data="QUJD")],
        )
        hm = build_user_message(cm)
        assert len(hm.images) == 1
        assert hm.images[0].base64_image == "QUJD"
        assert hm.images[0].mime_type == "image/jpeg"

    def test_text_only_message_has_no_image_content(self):
        hm = build_user_message(ChatMessage(role="user", content="hello"))
        assert hm.images == []

    def test_images_dropped_when_supports_images_false(self):
        cm = ChatMessage(
            role="user",
            content="describe",
            images=[ImageAttachment(media_type="image/jpeg", data="QUJD")],
        )
        hm = build_user_message(cm, supports_images=False)
        assert hm.images == []
        assert hm.text == "describe"


class TestPersistenceRoundTrip:
    def test_images_survive_model_dump_and_validate(self):
        cm = ChatMessage(
            role="user",
            content="x",
            images=[ImageAttachment(media_type="image/png", data="ZZ")],
        )
        dumped = cm.model_dump()
        assert dumped["images"] == [{"media_type": "image/png", "data": "ZZ", "id": ""}]
        assert ChatMessage.model_validate(dumped).images == cm.images
