from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class StreamEvent(BaseModel):
    event_type: Any


class MessageStartEvent(StreamEvent):
    event_type: Literal["message_start"] = "message_start"
    message_id: str | None = None


class TextChunkEvent(StreamEvent):
    event_type: Literal["text_chunk"] = "text_chunk"
    content: str


class ReasoningChunkEvent(StreamEvent):
    event_type: Literal["reasoning_chunk"] = "reasoning_chunk"
    content: str


class DataEvent(StreamEvent):
    event_type: Literal["data"] = "data"
    data_type: str
    data: dict[str, Any]


class ToolCallStartEvent(StreamEvent):
    event_type: Literal["tool_call_start"] = "tool_call_start"
    tool_call_id: str
    tool_name: str
    agent: str | None = None
    handoff: str | None = None


class ToolInputChunkEvent(StreamEvent):
    event_type: Literal["tool_input_chunk"] = "tool_input_chunk"
    tool_call_id: str
    input_chunk: str


class ToolInputCompleteEvent(StreamEvent):
    event_type: Literal["tool_input_complete"] = "tool_input_complete"
    tool_call_id: str
    tool_name: str
    tool_input: dict[str, Any] | str
    agent: str | None = None
    handoff: str | None = None


class ToolOutputEvent(StreamEvent):
    event_type: Literal["tool_output"] = "tool_output"
    tool_call_id: str
    tool_output: dict[str, Any]


class SuggestionEvent(StreamEvent):
    event_type: Literal["suggestion"] = "suggestion"
    suggestions: list[str]


class TextEndEvent(StreamEvent):
    """The answer's text is complete; the message may still get citations before it ends."""

    event_type: Literal["text_end"] = "text_end"


class MessageEndEvent(StreamEvent):
    event_type: Literal["message_end"] = "message_end"
    finish_reason: str | None = None


class ErrorEvent(StreamEvent):
    event_type: Literal["error"] = "error"
    error_code: str = "unknown"
    ref: str = ""


class StreamEndEvent(StreamEvent):
    event_type: Literal["stream_end"] = "stream_end"
