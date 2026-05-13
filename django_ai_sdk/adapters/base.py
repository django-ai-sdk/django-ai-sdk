from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from types import CoroutineType

    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.events import StreamEvent


class BasePipelineAdapter(ABC):
    """
    Base adapter interface for pipeline integration.

    Adapters convert their native streaming formats to normalized events
    that can be consumed by any protocol handler.
    """

    # Common attributes all adapters should have
    model: str | None = None
    instructions: str | None = None
    query: str | None = None

    # Message processing configuration
    merge_messages: bool = False

    def __init__(self) -> None:
        self.message_result: ChatMessage | None = None
        self._rag_sources: list[dict] = []

    @abstractmethod
    def get_messages(self, messages: list[ChatMessage]) -> list[dict] | list:
        """
        Convert internal ChatMessage format to pipeline-specific format.

        Args:
            messages: List of internal ChatMessage objects

        Returns:
            Pipeline-specific message format
        """
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[ChatMessage],
    ) -> CoroutineType[Any, Any, AsyncGenerator[StreamEvent, None]]:
        """
        Convert internal messages to pipeline format,
        del the pipeline,
        convert pipeline output to events.

        Args:
            messages: List of internal ChatMessage objects

        Yields:
            StreamEvent objects representing pipeline output
        """
        ...

    # TODO: this is unused at the moment, but I keep it as reference.
    # I want to refactor this, because we might want to alter the final result.
    # This alteration like: adding sources our, adding custom results, are
    # sometimes not really easy to handle in the stream context.
    # But when provided after the message is built, it's easier to modify the result.
    def get_message_result(self) -> ChatMessage | None:
        """Get the complete message result."""
        return self.message_result
