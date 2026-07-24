from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Literal, cast

from django.conf import settings
from pydantic import BaseModel, Field

from django_ai_sdk.common import ChatMessage, ImageAttachment
from django_ai_sdk.logger import get_logger
from django_ai_sdk.protocols.base import BaseProtocolHandler
from django_ai_sdk.protocols.utils import format_sse

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from django_ai_sdk.adapters.protocols import Streamable
    from django_ai_sdk.events import (
        DataEvent,
        ErrorEvent,
        MessageEndEvent,
        MessageStartEvent,
        ReasoningChunkEvent,
        SourceEvent,
        StreamEvent,
        SuggestionEvent,
        TextChunkEvent,
        ToolCallStartEvent,
        ToolInputCompleteEvent,
        ToolOutputEvent,
    )

logger = get_logger(__name__)


def _parse_image_data_url(url: str | None) -> tuple[str, str] | None:
    """Split a ``data:<mime>;base64,<payload>`` image URL into (media_type, base64).

    Returns None for non-data / non-image / non-base64 URLs (e.g. remote http
    URLs), which the caller skips — we only carry inline base64 images.
    """
    if not url or not url.startswith("data:"):
        return None
    try:
        header, payload = url[len("data:") :].split(",", 1)
    except ValueError:
        return None
    media_type, *params = header.split(";")
    # base64 need not be the last param (e.g. "image/jpeg;charset=utf-8;base64").
    if "base64" not in params or not media_type.startswith("image/"):
        return None
    return media_type, payload


# Defensive caps on inline images. Both accept an int (limit) or None (no limit),
# overridable via Django settings for consumers that want to raise/lower them.
DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MiB per image (decoded)
DEFAULT_MAX_IMAGES_PER_MESSAGE = 10


def _estimate_base64_bytes(payload: str) -> int:
    """Approximate the decoded byte size of a base64 payload (4 chars -> 3 bytes)."""
    return len(payload) * 3 // 4


# === Base Schema Classes ===


class Schema(BaseModel):
    """Base schema class with populate_by_name configuration."""

    model_config = {"populate_by_name": True}


# === Core Protocol Parts ===


class MessageStartPart(Schema):
    """Indicates the beginning of a new message with metadata."""

    type: Literal["start"] = "start"
    message_id: str = Field(
        validation_alias="message_id",
        serialization_alias="messageId",
    )


# === Text Parts ===


class TextStartPart(Schema):
    """Indicates the beginning of a text block."""

    type: Literal["text-start"] = "text-start"
    id: str


class TextDeltaPart(Schema):
    """Contains incremental text content for the text block."""

    type: Literal["text-delta"] = "text-delta"
    id: str
    delta: str


class TextEndPart(Schema):
    """Indicates the completion of a text block."""

    type: Literal["text-end"] = "text-end"
    id: str


# === Reasoning Parts ===


class ReasoningStartPart(Schema):
    """Indicates the beginning of a reasoning block."""

    type: Literal["reasoning-start"] = "reasoning-start"
    id: str


class ReasoningDeltaPart(Schema):
    """Contains incremental reasoning content for the reasoning block."""

    type: Literal["reasoning-delta"] = "reasoning-delta"
    id: str
    delta: str


class ReasoningEndPart(Schema):
    """Indicates the completion of a reasoning block."""

    type: Literal["reasoning-end"] = "reasoning-end"
    id: str


# === Source Parts ===


class SourceUrlPart(Schema):
    """References to external URLs."""

    type: Literal["source-url"] = "source-url"
    source_id: str = Field(validation_alias="source_id", serialization_alias="sourceId")
    url: str


class SourceDocumentPart(Schema):
    """References to documents or files."""

    type: Literal["source-document"] = "source-document"
    source_id: str = Field(validation_alias="source_id", serialization_alias="sourceId")
    media_type: str = Field(validation_alias="media_type", serialization_alias="mediaType")
    title: str


# === File Part ===


class FilePart(Schema):
    """References to files with their media type."""

    type: Literal["file"] = "file"
    url: str
    media_type: str = Field(validation_alias="media_type", serialization_alias="mediaType")


# === Data Parts ===


class DataPart(Schema):
    """Custom data parts for streaming arbitrary structured data."""

    type: str  # Must start with "data-" (e.g., "data-weather")
    data: dict[str, Any]


# === Error Part ===


class ErrorPart(Schema):
    """Error parts appended to the message as they are received."""

    type: Literal["error"] = "error"
    error_text: str = Field(validation_alias="error_text", serialization_alias="errorText")


# === Tool Parts ===


class ToolInputStartPart(Schema):
    """Indicates the beginning of tool input streaming."""

    type: Literal["tool-input-start"] = "tool-input-start"
    tool_call_id: str = Field(validation_alias="tool_call_id", serialization_alias="toolCallId")
    tool_name: str = Field(validation_alias="tool_name", serialization_alias="toolName")


class ToolInputDeltaPart(Schema):
    """Incremental chunks of tool input as it's being generated."""

    type: Literal["tool-input-delta"] = "tool-input-delta"
    tool_call_id: str = Field(validation_alias="tool_call_id", serialization_alias="toolCallId")
    input_text_delta: str = Field(
        validation_alias="input_text_delta",
        serialization_alias="inputTextDelta",
    )


class ToolInputAvailablePart(Schema):
    """Indicates that tool input is complete and ready for execution."""

    type: Literal["tool-input-available"] = "tool-input-available"
    tool_call_id: str = Field(validation_alias="tool_call_id", serialization_alias="toolCallId")
    tool_name: str = Field(validation_alias="tool_name", serialization_alias="toolName")
    input: dict[str, Any]


class ToolOutputAvailablePart(Schema):
    """Contains the result of tool execution."""

    type: Literal["tool-output-available"] = "tool-output-available"
    tool_call_id: str = Field(validation_alias="tool_call_id", serialization_alias="toolCallId")
    output: dict[str, Any]


# === Step Parts ===


class StartStepPart(Schema):
    """Indicates the start of a step."""

    type: Literal["start-step"] = "start-step"


class FinishStepPart(Schema):
    """Indicates that a step has been completed."""

    type: Literal["finish-step"] = "finish-step"


# === Completion Parts ===


class FinishPart(Schema):
    """Indicates the completion of a message."""

    type: Literal["finish"] = "finish"
    finishReason: str | None = None


class AbortPart(Schema):
    """Indicates the stream was aborted."""

    type: Literal["abort"] = "abort"
    reason: str


class DonePart(Schema):
    """Special marker for stream termination."""

    type: Literal["done"] = "done"


# === Union Type for All Stream Chunks ===

StreamChunk = (
    MessageStartPart
    | TextStartPart
    | TextDeltaPart
    | TextEndPart
    | ReasoningStartPart
    | ReasoningDeltaPart
    | ReasoningEndPart
    | SourceUrlPart
    | SourceDocumentPart
    | FilePart
    | DataPart
    | ErrorPart
    | ToolInputStartPart
    | ToolInputDeltaPart
    | ToolInputAvailablePart
    | ToolOutputAvailablePart
    | StartStepPart
    | FinishStepPart
    | FinishPart
    | AbortPart
    | DonePart
)


class VercelProtocolHandler(BaseProtocolHandler):
    """Converts normalized events to Vercel AI SDK Data Stream Protocol."""

    def __init__(self) -> None:
        self.message_id: str | None = None
        self.text_id: str | None = None
        self.text_started: bool = False
        # Reasoning state tracking
        self.reasoning_id: str | None = None
        self.reasoning_started: bool = False

    def to_chat_messages(
        self,
        protocol_messages: list[Any],
    ) -> list[ChatMessage]:
        """Convert Vercel Message objects to internal ChatMessage format."""
        max_bytes = getattr(settings, "AI_SDK_MAX_IMAGE_BYTES", DEFAULT_MAX_IMAGE_BYTES)
        max_count = getattr(
            settings, "AI_SDK_MAX_IMAGES_PER_MESSAGE", DEFAULT_MAX_IMAGES_PER_MESSAGE
        )

        messages = []
        for msg in protocol_messages:
            # Extract text content from parts using list comprehension
            content = " ".join(part.text for part in msg.parts if part.type == "text" and part.text)

            # Collect images from inline-base64 file parts (data: URLs). Remote
            # URLs are deliberately ignored: fetching a client-supplied URL
            # server-side would be an SSRF vector.
            images: list[ImageAttachment] = []
            for part in msg.parts:
                if part.type not in ("file", "image"):
                    continue
                parsed = _parse_image_data_url(getattr(part, "url", None))
                if parsed is None:
                    continue
                media_type, data = parsed
                if max_count is not None and len(images) >= max_count:
                    logger.warning(
                        "Dropping image(s): message exceeds AI_SDK_MAX_IMAGES_PER_MESSAGE (%d).",
                        max_count,
                    )
                    break
                if max_bytes is not None and _estimate_base64_bytes(data) > max_bytes:
                    logger.warning(
                        "Dropping image (~%d bytes): exceeds AI_SDK_MAX_IMAGE_BYTES (%d).",
                        _estimate_base64_bytes(data),
                        max_bytes,
                    )
                    continue
                part_media_type = getattr(part, "media_type", None)
                images.append(ImageAttachment(media_type=part_media_type or media_type, data=data))

            # Keep image-only messages (empty text) — the image is the payload.
            if content or images:
                messages.append(ChatMessage(role=msg.role, content=content, images=images))

        return messages

    def from_chat_messages(self, chat_messages: list[ChatMessage]) -> list[dict]:
        """Convert ChatMessage objects to Vercel AI SDK message format."""
        result = []
        for chat_message in chat_messages:
            parts = []

            if chat_message.content:
                parts.append({"type": "text", "text": chat_message.content})

            # Re-emit images as inline file parts so reloaded threads render them.
            # Skip any whose bytes could not be resolved (empty data).
            for image in chat_message.images:
                if not image.data:
                    continue
                parts.append(
                    {
                        "type": "file",
                        "mediaType": image.media_type,
                        "url": f"data:{image.media_type};base64,{image.data}",
                    }
                )

            if chat_message.sources:
                for source in chat_message.sources:
                    parts.append(
                        {
                            "type": "source-document",
                            "sourceId": source.get("source_id") or str(source.get("index", "")),
                            "mediaType": "file",
                            "title": source.get("title", ""),
                        }
                    )

            if chat_message.tool_calls:
                for tool_call in chat_message.tool_calls:
                    parts.append(
                        {
                            "type": f"tool-{tool_call.get('name', 'unknown')}",
                            "toolCallId": tool_call.get("id"),
                            "state": "output-available",
                            "input": tool_call.get("arguments", {}),
                            "output": tool_call.get("result", {}),
                        }
                    )

            result.append(
                {
                    "id": chat_message.id,
                    "role": chat_message.role,
                    "parts": parts,
                    "finish_reason": chat_message.finish_reason,
                    "tool_calls": chat_message.tool_calls,
                    "processing_time_ms": chat_message.processing_time_ms,
                    "has_errors": chat_message.has_errors,
                    "feedbacks": chat_message.metadata.get("feedbacks", []),
                    "created_at": chat_message.created_at,
                }
            )
        return result

    async def sse(
        self,
        adapter: Streamable,
        messages: list[ChatMessage],
    ) -> AsyncGenerator[bytes, None]:
        """Generate SSE-formatted streaming response from normalized events."""

        # Reset state for new stream
        self.text_started = False
        self.text_id = None
        self.message_id = None
        # Reset reasoning state
        self.reasoning_started = False
        self.reasoning_id = None

        events = adapter.stream(messages)
        protocol_stream = self.handle_stream(events)

        try:
            async for chunk in protocol_stream:
                # TODO: needs better handling for DONE termination
                if isinstance(chunk, DonePart):
                    yield format_sse("[DONE]")
                else:
                    yield format_sse(chunk.model_dump(exclude_none=True, by_alias=True))
        finally:
            await protocol_stream.aclose()
            # Ensure any open blocks are closed if stream was interrupted
            cleanup_chunks = []
            reasoning_id = self.reasoning_id
            text_id = self.text_id
            if self.reasoning_started and reasoning_id:
                cleanup_chunks.append(ReasoningEndPart(id=reasoning_id))
                self.reasoning_started = False
            if self.text_started and text_id:
                cleanup_chunks.append(TextEndPart(id=text_id))
                self.text_started = False

            # Yield cleanup chunks
            for chunk in cleanup_chunks:
                yield format_sse(chunk.model_dump(exclude_none=True, by_alias=True))

    async def handle_stream(
        self, events: AsyncGenerator[StreamEvent, None]
    ) -> AsyncGenerator[StreamChunk, None]:
        """Convert normalized events to Vercel StreamChunk objects."""

        async for event in events:
            match event.event_type:
                case "message_start":
                    # Use message_id from event (must be provided by adapter)
                    msg_event = cast("MessageStartEvent", event)
                    if not msg_event.message_id:
                        raise ValueError("message_id is required in MessageStartEvent")
                    self.message_id = msg_event.message_id
                    yield MessageStartPart(message_id=self.message_id)

                case "reasoning_chunk":
                    reasoning_event = cast("ReasoningChunkEvent", event)
                    # Start reasoning block if not already started
                    if not self.reasoning_started:
                        self.reasoning_id = str(uuid.uuid4())
                        yield ReasoningStartPart(id=self.reasoning_id)
                        self.reasoning_started = True

                    if self.reasoning_id:
                        yield ReasoningDeltaPart(
                            id=self.reasoning_id, delta=reasoning_event.content
                        )

                case "text_chunk":
                    text_event = cast("TextChunkEvent", event)
                    # Close reasoning block before starting text (if reasoning was open)
                    if self.reasoning_started and self.reasoning_id:
                        yield ReasoningEndPart(id=self.reasoning_id)
                        self.reasoning_started = False
                        self.reasoning_id = None

                    if not self.text_started:
                        self.text_id = str(uuid.uuid4())
                        yield TextStartPart(id=self.text_id)
                        self.text_started = True

                    if self.text_id:
                        yield TextDeltaPart(id=self.text_id, delta=text_event.content)

                case "tool_call_start":
                    tool_start_event = cast("ToolCallStartEvent", event)
                    yield ToolInputStartPart(
                        tool_call_id=tool_start_event.tool_call_id,
                        tool_name=tool_start_event.tool_name,
                    )

                case "tool_input_complete":
                    tool_input_event = cast("ToolInputCompleteEvent", event)
                    tool_input = (
                        tool_input_event.tool_input
                        if isinstance(tool_input_event.tool_input, dict)
                        else {"input": tool_input_event.tool_input}
                    )
                    yield ToolInputAvailablePart(
                        tool_call_id=tool_input_event.tool_call_id,
                        tool_name=tool_input_event.tool_name,
                        input=tool_input,
                    )

                case "tool_output":
                    tool_output_event = cast("ToolOutputEvent", event)
                    yield ToolOutputAvailablePart(
                        tool_call_id=tool_output_event.tool_call_id,
                        output=tool_output_event.tool_output,
                    )

                case "data":
                    # Convert intermediate DataEvent to Vercel DataPart
                    data_event = cast("DataEvent", event)
                    yield DataPart(type=f"data-{data_event.data_type}", data=data_event.data)

                case "suggestion":
                    suggestion_event = cast("SuggestionEvent", event)
                    yield DataPart(
                        type="data-suggestions", data={"suggestions": suggestion_event.suggestions}
                    )

                case "source":
                    src = cast("SourceEvent", event)
                    yield SourceDocumentPart(
                        source_id=src.source_id,
                        media_type=src.media_type,
                        title=src.title,
                    )

                case "error":
                    error_event = cast("ErrorEvent", event)
                    yield ErrorPart(error_text=error_event.error_message)

                case "message_end":
                    end_event = cast("MessageEndEvent", event)
                    # Close reasoning block if still open
                    if self.reasoning_started and self.reasoning_id:
                        yield ReasoningEndPart(id=self.reasoning_id)
                        self.reasoning_started = False
                        self.reasoning_id = None
                    if self.text_started and self.text_id:
                        yield TextEndPart(id=self.text_id)
                        self.text_started = False
                    # Default to "stop" if no finish_reason provided
                    finish_reason = end_event.finish_reason or "stop"
                    yield FinishPart(finishReason=finish_reason)

                case "stream_end":
                    yield DonePart()
