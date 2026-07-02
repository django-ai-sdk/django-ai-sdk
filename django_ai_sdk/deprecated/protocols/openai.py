from __future__ import annotations

import json
import time
import uuid
from typing import TYPE_CHECKING, Any, Literal, cast

from django_ai_sdk.common import ChatMessage
from django_ai_sdk.protocols.base import BaseProtocolHandler
from django_ai_sdk.protocols.utils import format_sse
from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from django_ai_sdk.adapters.protocols import Streamable
    from django_ai_sdk.events import (
        ErrorEvent,
        MessageEndEvent,
        TextChunkEvent,
        ToolCallStartEvent,
        ToolInputCompleteEvent,
    )


# OpenAI chunk schemas
class OpenAIDelta(BaseModel):
    """Delta object for streaming chunks."""

    role: str | None = None
    content: str | None = None
    tool_calls: list[dict] | None = None


class OpenAIChoice(BaseModel):
    """Choice object for streaming chunks."""

    index: int = 0
    delta: OpenAIDelta
    finish_reason: str | None = None


class OpenAIStreamChunk(BaseModel):
    """OpenAI streaming chunk format."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[OpenAIChoice]


class OpenAIProtocolHandler(BaseProtocolHandler):
    """
    Convert StreamEvent to OpenAI-compatible SSE format
    """

    def __init__(self, model: str = "gpt-4") -> None:
        self.model = model
        self.id: str | None = None
        self.text_buffer: str = ""
        self.tool_calls_buffer: list[dict] = []
        self.current_tool_call: dict | None = None

    def to_chat_messages(
        self,
        protocol_messages: list[Any],
    ) -> list[ChatMessage]:
        """Convert OpenAI format messages to internal ChatMessage."""
        messages = []
        for msg in protocol_messages:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages.append(
                    ChatMessage(
                        role=msg["role"],
                        content=msg["content"],
                    )
                )
        return messages

    def from_chat_messages(self, chat_messages: list[ChatMessage]) -> list[dict]:
        """Convert ChatMessage back to OpenAI format."""
        return [{"role": msg.role, "content": msg.content} for msg in chat_messages]

    def _create_chunk(
        self,
        delta: OpenAIDelta,
        finish_reason: str | None = None,
    ) -> OpenAIStreamChunk:
        """Create an OpenAI stream chunk."""
        return OpenAIStreamChunk(
            id=self.id or f"chatcmpl-{uuid.uuid4().hex[:24]}",
            created=int(time.time()),
            model=self.model,
            choices=[OpenAIChoice(delta=delta, finish_reason=finish_reason)],
        )

    async def sse(
        self,
        adapter: Streamable,
        messages: list[ChatMessage],
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate SSE-formatted streaming response in OpenAI format.
        """
        # Generate ID
        self.id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        # Emit start chunk with role
        start_chunk = self._create_chunk(OpenAIDelta(role="assistant"))
        yield format_sse(start_chunk.model_dump())

        # Get events from adapter
        events = adapter.stream(messages)

        try:
            async for event in events:
                match event.event_type:
                    case "message_start":
                        # Message already started, continue
                        pass

                    case "text_chunk":
                        text_event = cast("TextChunkEvent", event)
                        # Emit content chunk immediately
                        chunk = self._create_chunk(OpenAIDelta(content=text_event.content))
                        yield format_sse(chunk.model_dump())

                    case "reasoning_chunk":
                        # Reasoning is internal, skip in OpenAI format
                        pass

                    case "tool_call_start":
                        tool_start = cast("ToolCallStartEvent", event)
                        # Start buffering tool call
                        self.current_tool_call = {
                            "id": tool_start.tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_start.tool_name,
                                "arguments": "",
                            },
                        }

                    case "tool_input_complete":
                        tool_input = cast("ToolInputCompleteEvent", event)
                        # Add arguments to current tool call
                        if self.current_tool_call:
                            # TODO: move into a utils, there should be a single way for
                            # formatting tool input arguments
                            args = (
                                json.dumps(tool_input.tool_input)
                                if isinstance(tool_input.tool_input, dict)
                                else str(tool_input.tool_input)
                            )
                            self.current_tool_call["function"]["arguments"] = args
                            self.tool_calls_buffer.append(self.current_tool_call)
                            self.current_tool_call = None

                    case "tool_output":
                        # Tool output happens after the assistant message
                        pass

                    case "data":
                        # Custom data events
                        pass

                    case "message_end":
                        end_event = cast("MessageEndEvent", event)
                        # Emit final chunk with finish_reason and any buffered tool calls
                        delta = OpenAIDelta()

                        if self.tool_calls_buffer:
                            delta.tool_calls = self.tool_calls_buffer

                        final_chunk = self._create_chunk(
                            delta=delta,
                            finish_reason=end_event.finish_reason or "stop",
                        )
                        yield format_sse(final_chunk.model_dump())

                    case "error":
                        error_event = cast("ErrorEvent", event)
                        # Note: error chunk are emitted differently from regular chunks
                        yield format_sse(
                            {
                                "error": {
                                    "message": error_event.error_message,
                                    "type": "server_error",
                                }
                            }
                        )

                    case "stream_end":
                        # Termination marker
                        yield format_sse("[DONE]")

        finally:
            # Cleanup if needed
            pass
