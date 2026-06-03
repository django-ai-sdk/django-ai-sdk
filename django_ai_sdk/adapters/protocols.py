from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.events import StreamEvent
    from django_ai_sdk.suggestions import SuggestionGenerator

T = TypeVar("T", bound=BaseModel)


class Runnable(Protocol):
    """An LLM runner"""

    model: str | None = None
    instructions: str | None = None

    async def run(
        self,
        messages: list[ChatMessage],
        system_prompt: str | None = None,
        response_format: type[T] | None = None,
    ) -> T | str | None: ...


class Streamable(Protocol):
    """Stream an LLM response as normalized events."""

    model: str | None = None
    instructions: str | None = None
    suggestion_generator: SuggestionGenerator | None = None

    def stream(
        self,
        messages: list[ChatMessage],
    ) -> AsyncGenerator[StreamEvent, None]: ...
