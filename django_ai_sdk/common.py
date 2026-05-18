import time
import traceback
from textwrap import dedent
from typing import Any, Literal, NewType

from pydantic import BaseModel, Field

from .logger import get_logger

logger = get_logger(__name__)


Prompt = NewType("Prompt", str)


def prompt(text: str) -> Prompt:
    """Helper to create Prompt type with dedented text."""
    return Prompt(dedent(text))


class ChatMessage(BaseModel):
    """Internal SDK representation of a chat message."""

    # Core message data
    role: Literal["system", "user", "assistant"]
    content: str = ""
    reasoning: str | None = None
    id: str = ""

    # Rich metadata
    tool_calls: list[dict] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)
    model: str = ""
    finish_reason: str = ""
    adapter_type: str = ""
    errors: list[str] = Field(default_factory=list)
    usage: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)

    # Timestamps & timing
    created_at: str = ""
    processing_time_ms: int = 0
    started_at: float = Field(default_factory=time.time)
    completed_at: float = 0

    # Helpers
    @property
    def text(self) -> str:
        """Text property for compatibility"""
        return self.content

    @property
    def duration(self) -> int:
        """Total processing time in milliseconds."""
        if self.completed_at > 0:
            return int((self.completed_at - self.started_at) * 1000)
        return 0

    @property
    def has_tools(self) -> bool:
        """Whether this message used tools."""
        return len(self.tool_calls) > 0

    @property
    def has_errors(self) -> bool:
        """Whether this message had errors."""
        return len(self.errors) > 0

    def finalize(self, finish_reason: str = "") -> None:
        """Mark message as complete."""
        self.finish_reason = finish_reason
        self.completed_at = time.time()
        self.processing_time_ms = self.duration


class MessageChunk(BaseModel):
    """Universal chunk format for storage and standardization."""

    type: str  # 'text', 'reasoning', 'tool_call_start', 'tool_input', 'tool_output', 'error'
    content: Any
    metadata: dict = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class StreamWriter:
    """Writes ChatMessage from streaming chunks."""

    def __init__(
        self,
        adapter_type: str,
        message_id: str,
        model: str = "",
        role: Literal["system", "user", "assistant"] = "assistant",
        storage_callback: Any = None,
    ) -> None:
        self.message = ChatMessage(
            id=message_id,
            role=role,
            adapter_type=adapter_type,
            model=model,
            started_at=time.time(),
        )
        self._pending_tool_calls = {}  # Track multiple tool calls by ID
        self.storage_callback = storage_callback

        logger.debug(
            f"StreamWriter initialized: adapter={adapter_type}, model={model}, role={role}, storage={'enabled' if storage_callback else 'disabled'}"
        )

    def add_chunk(self, chunk: MessageChunk) -> ChatMessage:
        """Process a chunk and update message."""
        logger.debug(
            f"Processing chunk: type={chunk.type}, content_length={len(str(chunk.content))}"
        )

        if chunk.type == "text":
            self.message.content += chunk.content
            logger.debug(f"Added text chunk, total content length now: {len(self.message.content)}")

        elif chunk.type == "reasoning":
            # Initialize reasoning field if first chunk
            if self.message.reasoning is None:
                self.message.reasoning = ""
            self.message.reasoning += chunk.content
            logger.debug(
                f"Added reasoning chunk, total reasoning length now: {len(self.message.reasoning or '')}"
            )

        elif chunk.type == "tool_call_start":
            tool_call_id = chunk.content["tool_call_id"]
            tool_name = chunk.content["tool_name"]
            self._pending_tool_calls[tool_call_id] = {
                "id": tool_call_id,
                "name": tool_name,
                "arguments": {},
                "result": None,
            }
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
                # Move completed tool call to message
                completed_tool = self._pending_tool_calls[tool_call_id]
                self.message.tool_calls.append(completed_tool)
                del self._pending_tool_calls[tool_call_id]
                logger.debug(f"Completed tool call: {completed_tool['name']} (ID: {tool_call_id})")
            else:
                logger.debug(f"Received tool output for unknown tool call ID: {tool_call_id}")

        elif chunk.type == "error":
            error_message = chunk.content["error_message"]
            self.message.errors.append(error_message)
            logger.debug(f"Added error to message: {error_message}")
        else:
            logger.debug(f"Unknown chunk type: {chunk.type}")

        return self.message

    async def finalize(self, finish_reason: str = "", usage: dict | None = None) -> ChatMessage:
        """Complete the message"""
        logger.debug(f"Finalizing message with reason: {finish_reason}")

        # Add any remaining pending tool calls (in casse tool_output never came)
        pending_tools_count = len(self._pending_tool_calls)
        if pending_tools_count > 0:
            logger.debug(f"Adding {pending_tools_count} pending tool calls to final message")
            for tool_call in self._pending_tool_calls.values():
                self.message.tool_calls.append(tool_call)
                logger.debug(
                    f"Added pending tool call: {tool_call['name']} (ID: {tool_call['id']})"
                )
        self._pending_tool_calls.clear()

        # Set usage if provided
        if usage:
            self.message.usage = usage
            logger.debug(f"Set usage on message: {usage}")

        # Finalize message
        self.message.finalize(finish_reason)
        logger.debug(
            f"Message finalized: content_length={len(self.message.content)}, tool_calls={len(self.message.tool_calls)}, errors={len(self.message.errors)}, duration={self.message.duration}ms"
        )

        # Auto-store if callback provided
        if self.storage_callback:
            logger.debug("Attempting to store message via storage callback")
            try:
                # Store the message based on connected adapter.
                await self.storage_callback(self.message)
                logger.debug("Message stored successfully via callback")
            except Exception as storage_error:
                # Log error but don't break streaming
                logger.error(f"Storage callback failed: {storage_error}")
                logger.error(f"Storage callback traceback: {traceback.format_exc()}")
        else:
            logger.debug("No storage callback provided, skipping message storage")

        return self.message
