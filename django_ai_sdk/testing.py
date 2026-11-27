from __future__ import annotations

from typing import TYPE_CHECKING, Any

from haystack.components.generators.chat import MockChatGenerator

if TYPE_CHECKING:
    from haystack.dataclasses import ChatMessage

__all__ = ["FakeLLM"]


class FakeLLM:
    """The last reply repeats once the others are used. With no replies, the
    generator echoes the last message."""

    def __init__(self, *replies: str | ChatMessage) -> None:
        self.replies = list(replies)
        self.calls: list[list[ChatMessage]] = []

    def __call__(self, **kwargs: Any) -> MockChatGenerator:
        return MockChatGenerator(
            response_fn=self._reply,
            model=kwargs.get("model") or "fake",
            streaming_callback=kwargs.get("streaming_callback"),
        )

    def _reply(self, messages: list[ChatMessage]) -> str | ChatMessage:
        self.calls.append(list(messages))
        if not self.replies:
            return next((m.text for m in reversed(messages) if m.text), "") or ""
        return self.replies[min(len(self.calls), len(self.replies)) - 1]
