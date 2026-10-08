from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class StreamEvent(BaseModel):
    """Base class for all normalized streaming events."""

    event_type: str  # Subclasses override with specific Literal types


class MessageStartEvent(StreamEvent):
    """Start of a new message."""

    event_type: Literal["message_start"] = "message_start"
    message_id: str | None = None


class TextChunkEvent(StreamEvent):
    """Incremental text content."""

    event_type: Literal["text_chunk"] = "text_chunk"
    content: str


class ReasoningChunkEvent(StreamEvent):
    """Incremental reasoning content (for models that support it)."""

    event_type: Literal["reasoning_chunk"] = "reasoning_chunk"
    content: str


class DataEvent(StreamEvent):
    """Custom data event for arbitrary structured data (e.g., RAG retrieval)."""

    event_type: Literal["data"] = "data"
    data_type: str  # The suffix after "data-", e.g., "rag-retrieval"
    data: dict[str, Any]


class ToolCallStartEvent(StreamEvent):
    """Beginning of a tool call."""

    event_type: Literal["tool_call_start"] = "tool_call_start"
    tool_call_id: str
    tool_name: str
    # `agent` for a subagent's own tool, `handoff` for the coordinator delegating to one.
    agent: str | None = None
    handoff: str | None = None


class ToolInputChunkEvent(StreamEvent):
    """Incremental tool input as it's being generated."""

    event_type: Literal["tool_input_chunk"] = "tool_input_chunk"
    tool_call_id: str
    input_chunk: str


class ToolInputCompleteEvent(StreamEvent):
    """Tool input is complete and ready for execution."""

    event_type: Literal["tool_input_complete"] = "tool_input_complete"
    tool_call_id: str
    tool_name: str
    tool_input: dict[str, Any] | str
    agent: str | None = None
    handoff: str | None = None


class ToolOutputEvent(StreamEvent):
    """Result of tool execution."""

    event_type: Literal["tool_output"] = "tool_output"
    tool_call_id: str
    tool_output: dict[str, Any]


class SuggestionEvent(StreamEvent):
    """Follow-up suggestions generated after a response."""

    event_type: Literal["suggestion"] = "suggestion"
    suggestions: list[str]


class MessageEndEvent(StreamEvent):
    """End of message."""

    event_type: Literal["message_end"] = "message_end"
    finish_reason: str | None = None


class ErrorEvent(StreamEvent):
    """Error during streaming, as a client may see it (see django_ai_sdk.errors)."""

    event_type: Literal["error"] = "error"
    error_code: str = "unknown"
    ref: str = ""


class StreamEndEvent(StreamEvent):
    """End of entire stream."""

    event_type: Literal["stream_end"] = "stream_end"
