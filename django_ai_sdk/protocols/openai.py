from __future__ import annotations

import copy
import json
import time
import uuid
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel

from django_ai_sdk.common import ChatMessage
from django_ai_sdk.errors import get_error_spec
from django_ai_sdk.protocols.base import BaseProtocolHandler
from django_ai_sdk.protocols.utils import format_sse

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from django_ai_sdk.adapters.interfaces import Streamable
    from django_ai_sdk.events import (
        ErrorEvent,
        MessageEndEvent,
        TextChunkEvent,
        ToolCallStartEvent,
        ToolInputCompleteEvent,
    )


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
        return [{"role": msg.role, "content": msg.content} for msg in chat_messages]

    def _create_chunk(
        self,
        delta: OpenAIDelta,
        finish_reason: str | None = None,
    ) -> OpenAIStreamChunk:
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
        state = copy.copy(self)
        state.id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        state.tool_calls_buffer = []
        state.current_tool_call = None

        start_chunk = state._create_chunk(OpenAIDelta(role="assistant"))
        yield format_sse(start_chunk.model_dump())

        events = adapter.stream(messages)

        try:
            async for event in events:
                match event.event_type:
                    case "message_start":
                        pass

                    case "text_chunk":
                        text_event = cast("TextChunkEvent", event)
                        chunk = state._create_chunk(OpenAIDelta(content=text_event.content))
                        yield format_sse(chunk.model_dump())

                    case "reasoning_chunk":
                        pass

                    case "tool_call_start":
                        tool_start = cast("ToolCallStartEvent", event)
                        state.current_tool_call = {
                            "id": tool_start.tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_start.tool_name,
                                "arguments": "",
                            },
                        }

                    case "tool_input_complete":
                        tool_input = cast("ToolInputCompleteEvent", event)
                        args = (
                            json.dumps(tool_input.tool_input)
                            if isinstance(tool_input.tool_input, dict)
                            else str(tool_input.tool_input)
                        )
                        if state.current_tool_call:
                            state.current_tool_call["function"]["arguments"] = args
                            state.tool_calls_buffer.append(state.current_tool_call)
                            state.current_tool_call = None
                        else:
                            for call in state.tool_calls_buffer:
                                if call["id"] == tool_input.tool_call_id:
                                    call["function"]["arguments"] = args

                    case "tool_output":
                        pass

                    case "data":
                        pass

                    case "message_end":
                        end_event = cast("MessageEndEvent", event)
                        delta = OpenAIDelta()

                        if state.tool_calls_buffer:
                            delta.tool_calls = state.tool_calls_buffer

                        final_chunk = state._create_chunk(
                            delta=delta,
                            # Events use "content-filter" (Vercel). OpenAI uses "content_filter".
                            finish_reason=(end_event.finish_reason or "stop").replace("-", "_"),
                        )
                        yield format_sse(final_chunk.model_dump())

                    case "error":
                        error_event = cast("ErrorEvent", event)
                        spec = get_error_spec(error_event.error_code)
                        yield format_sse(
                            {
                                "error": {
                                    "message": str(spec.message),
                                    "type": "server_error",
                                    "code": spec.code,
                                    "retryable": spec.retryable,
                                    "ref": error_event.ref,
                                }
                            }
                        )

                    case "stream_end":
                        yield format_sse("[DONE]")

        finally:
            pass
