import uuid
from collections.abc import AsyncGenerator
from typing import Any, Literal, cast

from pydantic import BaseModel, Field

from django_ai_sdk.adapters.base import BasePipelineAdapter
from django_ai_sdk.common import ChatMessage
from django_ai_sdk.events import (
    DataEvent,
    ErrorEvent,
    MessageEndEvent,
    MessageStartEvent,
    ReasoningChunkEvent,
    StreamEvent,
    TextChunkEvent,
    ToolCallStartEvent,
    ToolInputCompleteEvent,
    ToolOutputEvent,
)
from django_ai_sdk.logger import get_logger
from django_ai_sdk.protocols.base import BaseProtocolHandler
from django_ai_sdk.protocols.utils import format_sse

logger = get_logger(__name__)

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
        messages = []
        for msg in protocol_messages:
            # Extract text content from parts using list comprehension
            content = " ".join(part.text for part in msg.parts if part.type == "text" and part.text)

            if content:
                messages.append(ChatMessage(role=msg.role, content=content))

        return messages

    def from_chat_messages(self, chat_messages: list[ChatMessage]) -> list[dict]:
        """Convert ChatMessage objects to Vercel AI SDK message format."""
        result = []
        for chat_message in chat_messages:
            parts = []

            if chat_message.content:
                parts.append({"type": "text", "text": chat_message.content})

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
                    "adapter_type": chat_message.adapter_type,
                    "finish_reason": chat_message.finish_reason,
                    "tool_calls": chat_message.tool_calls,
                    "processing_time_ms": chat_message.processing_time_ms,
                    "has_errors": chat_message.has_errors,
                    "usage": chat_message.usage,
                }
            )
            logger.debug(f"Converting message {chat_message.id}: usage={chat_message.usage}")

        return result

    async def sse(  # type: ignore
        self,
        adapter: BasePipelineAdapter,
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

        events = adapter.stream(messages)  # type: ignore
        protocol_stream = self.handle_stream(events)  # type: ignore

        try:
            async for chunk in protocol_stream:
                # TODO: needs better handling for DONE termination
                if isinstance(chunk, DonePart):
                    yield format_sse("[DONE]")
                else:
                    yield format_sse(chunk.model_dump(exclude_none=True, by_alias=True))
        finally:
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
                    yield ToolInputAvailablePart(
                        tool_call_id=tool_input_event.tool_call_id,
                        tool_name=tool_input_event.tool_name,
                        input=tool_input_event.tool_input,
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
                    logger.info(f"Message end: usage={end_event.usage}")
                    yield FinishPart(finishReason=finish_reason)
                    # Emit usage as custom data event if available
                    if end_event.usage:
                        logger.info(f"Emitting data-usage event: {end_event.usage}")
                        yield DataPart(type="data-usage", data=end_event.usage)
                    else:
                        logger.info(
                            "No usage in message end event - data-usage event will NOT be emitted"
                        )

                case "stream_end":
                    yield DonePart()
