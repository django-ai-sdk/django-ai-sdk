"""Test helpers: an agent that answers without a model.

`FakeLLM` is a generator factory, so it goes where an agent's `llm` goes:

    from django_ai_sdk.testing import FakeLLM

    async def test_a_question_is_answered():
        fake = FakeLLM("On its way.")
        with patch.object(SupportAgent, "llm", fake):
            assert await SupportAgent().run([...]) == "On its way."
        assert fake.calls  # one list of messages per call

The generators stream like real ones, so chat pipelines work too. They never
call a service.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from haystack.components.generators.chat import MockChatGenerator

if TYPE_CHECKING:
    from haystack.dataclasses import ChatMessage

__all__ = ["FakeLLM"]


class FakeLLM:
    """Generator factory whose generators answer with `replies`, in order.

    The last reply repeats once the others are used. With no replies, the
    generator echoes the last message. Pass an assistant `ChatMessage` to reply
    with a tool call.

    `calls` records the messages of every call, across all generators.
    """

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
