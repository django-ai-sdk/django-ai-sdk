"""
Base protocol handling with handler selection.

This module provides the main interface for selecting and using
different protocol handlers based on the target format.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any, cast

from django_ai_sdk.adapters.protocols import Streamable
from django_ai_sdk.common import ChatMessage


class BaseProtocolHandler(ABC):
    """Base class for protocol handlers."""

    @abstractmethod
    def to_chat_messages(
        self,
        protocol_messages: list[Any],
    ) -> list[ChatMessage]:
        """Convert protocol-specific messages to internal ChatMessage format."""
        pass

    @abstractmethod
    def from_chat_messages(self, chat_messages: list[ChatMessage]) -> list[dict[str, Any]]:
        """Convert internal ChatMessages to protocol-specific message format."""
        pass

    @abstractmethod
    async def sse(
        self,
        adapter: Streamable,
        messages: list[ChatMessage],
    ) -> AsyncGenerator[bytes, None]:
        """Generate SSE-formatted streaming response from normalized events."""
        yield cast("bytes", None)  # pragma: no cover
