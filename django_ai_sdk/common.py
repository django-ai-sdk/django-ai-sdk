from __future__ import annotations

import time
import traceback
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Literal, NewType

from pydantic import BaseModel, Field

from .logger import get_logger
from .receipts import receipt

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

logger = get_logger(__name__)


Prompt = NewType("Prompt", str)

THREAD_TITLE_MAX_LENGTH = 255


def prompt(text: str) -> Prompt:
    return Prompt(dedent(text))


class Attachment(BaseModel):
    document_id: str
    media_type: str = ""
    filename: str = ""
    memory_id: str = ""

    url: str = Field(default="", exclude=True)
    context: str = Field(default="", exclude=True)
    data: str = Field(default="", exclude=True)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = ""
    reasoning: str | None = None
    id: str = ""
    attachments: list[Attachment] = Field(default_factory=list)

    tool_calls: list[dict] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)
    model: str = ""
    finish_reason: str = ""
    errors: list[str] = Field(default_factory=list)
    error_code: str = ""
    error_ref: str = ""
    metadata: dict = Field(default_factory=dict)

    created_at: str = ""
    processing_time_ms: int = 0
    started_at: float = Field(default_factory=time.time)
    completed_at: float = 0

    @property
    def text(self) -> str:
        return self.content

    @property
    def duration(self) -> int:
        """Total processing time in milliseconds."""
        if self.completed_at > 0:
            return int((self.completed_at - self.started_at) * 1000)
        return 0

    @property
    def has_tools(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def finalize(self, finish_reason: str = "") -> None:
        self.finish_reason = finish_reason
        self.completed_at = time.time()
        self.processing_time_ms = self.duration


def as_chat_messages(messages: Iterable[ChatMessage | Mapping[str, Any]]) -> list[ChatMessage]:
    return [
        m if isinstance(m, ChatMessage) else ChatMessage.model_validate(dict(m)) for m in messages
    ]


def last_user_message(messages: list[ChatMessage]) -> ChatMessage | None:
    return next((m for m in reversed(messages) if m.role == "user"), None)


def last_user_text(messages: list[ChatMessage]) -> str:
    message = last_user_message(messages)
    return (message.content if message is not None else "") or ""


class MessageChunk(BaseModel):
    type: str
    content: Any
    metadata: dict = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class StreamWriter:
    def __init__(
        self,
        message_id: str,
        model: str = "",
        role: Literal["system", "user", "assistant"] = "assistant",
        storage_callback: Any = None,
    ) -> None:
        self.message = ChatMessage(
            id=message_id,
            role=role,
            model=model,
            started_at=time.time(),
        )
        self._pending_tool_calls = {}
        self.storage_callback = storage_callback
        self._content_chunks: list[str] = []
        self._reasoning_chunks: list[str] = []

        logger.debug(
            f"StreamWriter initialized: model={model}, role={role}, storage={'enabled' if storage_callback else 'disabled'}"
        )

    def add_chunk(self, chunk: MessageChunk) -> ChatMessage:
        if chunk.type == "text":
            self._content_chunks.append(chunk.content)

        elif chunk.type == "reasoning":
            self._reasoning_chunks.append(chunk.content)
            logger.debug(
                f"Added reasoning chunk, total reasoning length now: {sum(len(c) for c in self._reasoning_chunks)}"
            )

        elif chunk.type == "tool_call_start":
            tool_call_id = chunk.content["tool_call_id"]
            if tool_call_id in self._pending_tool_calls:
                return self.message
            tool_name = chunk.content["tool_name"]
            tool_call = {
                "id": tool_call_id,
                "name": tool_name,
                "arguments": {},
                "result": None,
            }
            for key in ("agent", "handoff"):
                if value := chunk.metadata.get(key):
                    tool_call[key] = value
            self._pending_tool_calls[tool_call_id] = tool_call
            self.message.tool_calls.append(tool_call)
            logger.debug(f"Started tool call: {tool_name} (ID: {tool_call_id})")

        elif chunk.type == "tool_input":
            tool_call_id = chunk.content["tool_call_id"]
            if tool_call_id in self._pending_tool_calls:
                self._pending_tool_calls[tool_call_id]["arguments"] = chunk.content["tool_input"]
                logger.debug(f"Added input for tool call {tool_call_id}")
            else:
                logger.debug(f"Received tool input for unknown tool call ID: {tool_call_id}")

        elif chunk.type == "tool_output":
            tool_call_id = chunk.content["tool_call_id"]
            if tool_call_id in self._pending_tool_calls:
                self._pending_tool_calls[tool_call_id]["result"] = chunk.content["tool_output"]
                del self._pending_tool_calls[tool_call_id]
                logger.debug(f"Completed tool call (ID: {tool_call_id})")
            else:
                logger.debug(f"Received tool output for unknown tool call ID: {tool_call_id}")

        elif chunk.type == "error":
            error_message = chunk.content["error_message"]
            self.message.errors.append(error_message)
            self.message.error_code = chunk.content.get("error_code", "")
            self.message.error_ref = chunk.content.get("error_ref", "")
            logger.debug(f"Added error to message: {error_message}")
        else:
            logger.debug(f"Unknown chunk type: {chunk.type}")

        return self.message

    @property
    def text(self) -> str:
        return "".join(self._content_chunks)

    def replace_text(self, text: str) -> None:
        self._content_chunks = [text]

    def drop_reasoning(self) -> None:
        self._reasoning_chunks = []

    def drop_tool_details(self, receipts: Mapping[str, tuple[str, ...]]) -> None:
        for call in self.message.tool_calls:
            kept = receipt(call["result"], receipts.get(call["name"]))
            call["arguments"] = {}
            call["result"] = {"result": kept} if kept else None

    async def finalize(self, finish_reason: str = "") -> ChatMessage:
        logger.debug(f"Finalizing message with reason: {finish_reason}")

        if self._content_chunks:
            self.message.content = "".join(self._content_chunks)
        if self._reasoning_chunks:
            self.message.reasoning = "".join(self._reasoning_chunks)

        pending_tools_count = len(self._pending_tool_calls)
        if pending_tools_count > 0:
            logger.debug(
                f"Finalizing {pending_tools_count} tool calls without output "
                f"(stored with result=None)"
            )
        self._pending_tool_calls.clear()

        self.message.finalize(finish_reason)
        logger.debug(
            f"Message finalized: content_length={len(self.message.content)}, tool_calls={len(self.message.tool_calls)}, errors={len(self.message.errors)}, duration={self.message.duration}ms"
        )

        if self.storage_callback:
            logger.debug("Attempting to store message via storage callback")
            try:
                await self.storage_callback(self.message)
                logger.debug("Message stored successfully via callback")
            except Exception as storage_error:
                logger.error(f"Storage callback failed: {storage_error}")
                logger.error(f"Storage callback traceback: {traceback.format_exc()}")
        else:
            logger.debug("No storage callback provided, skipping message storage")

        return self.message
