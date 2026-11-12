"""The Run adapter: one-off model calls and headless tool loops, as Haystack Agent runs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage as HaystackChatMessage
from haystack.dataclasses import ImageContent
from pydantic import ValidationError

from django_ai_sdk.generators import schema_kwargs
from django_ai_sdk.logger import get_logger

if TYPE_CHECKING:
    from django_ai_sdk.adapters.interfaces import T
    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.oversight.agent_run import OpenTurn

logger = get_logger(__name__)


def get_user_text(message: ChatMessage) -> str:
    """User text followed by the context line of each attachment."""
    hints = [a.context for a in message.attachments if a.context]
    return "\n\n".join([message.content, *hints]) if hints else message.content


def get_user_message(message: ChatMessage) -> HaystackChatMessage:
    """Haystack user message, multimodal when images were inlined for a vision model."""
    text = get_user_text(message)
    images = [
        ImageContent(base64_image=a.data, mime_type=a.media_type)
        for a in message.attachments
        if a.data
    ]
    if not images:
        return HaystackChatMessage.from_user(text)
    return HaystackChatMessage.from_user(content_parts=[text, *images] if text else images)


class Run:
    """
    Runnable Haystack adapter.
    """

    model: str | None = None
    instructions: str | None = None

    def __init__(
        self,
        generator: Any,
        *,
        model: str | None = None,
        instructions: str | None = None,
    ) -> None:
        self.generator = generator
        self.model = model
        self.instructions = instructions

    def get_messages(self, messages: list[ChatMessage]) -> list[HaystackChatMessage]:
        """Quick conversion."""
        conversation = [m for m in messages if m.role in ("user", "assistant")]
        converted: list[HaystackChatMessage] = []
        for msg in conversation:
            if msg.role == "user":
                converted.append(get_user_message(msg))
            elif msg.role == "assistant":
                converted.append(HaystackChatMessage.from_assistant(msg.content))  # type: ignore[arg-type]
        return converted

    @overload
    async def run(
        self,
        messages: list[ChatMessage],
        system_prompt: str | None = None,
        *,
        response_format: None = None,
        tools: list[Any] | None = None,
        max_agent_steps: int | None = None,
        hooks: dict[str, list[Any]] | None = None,
        turn: OpenTurn | None = None,
    ) -> str | None: ...
    @overload
    async def run(
        self,
        messages: list[ChatMessage],
        system_prompt: str | None = None,
        *,
        response_format: type[T],
        tools: list[Any] | None = None,
        max_agent_steps: int | None = None,
        hooks: dict[str, list[Any]] | None = None,
        turn: OpenTurn | None = None,
    ) -> T | None: ...

    async def run(
        self,
        messages: list[ChatMessage],
        system_prompt: str | None = None,
        response_format: type[T] | None = None,
        *,
        tools: list[Any] | None = None,
        max_agent_steps: int | None = None,
        hooks: dict[str, list[Any]] | None = None,
        turn: OpenTurn | None = None,
    ) -> T | str | None:
        """Run the generator as a Haystack Agent: once, or as a tool loop with `tools`.

        Oversight runs inside it as hooks (see `oversight.runs`). `turn` is the turn
        this run answers: its question is gated and its answer finished in the run.
        Raises `Blocked` when either refuses.
        """
        from django_ai_sdk.oversight.runs import run_attached

        def to_haystack(given: list[ChatMessage]) -> list[HaystackChatMessage]:
            converted = self.get_messages(given)
            # A message, not the Agent's system_prompt, which Haystack renders as a template.
            return (
                [HaystackChatMessage.from_system(system_prompt), *converted]
                if system_prompt
                else converted
            )

        steps = {"max_agent_steps": max_agent_steps} if max_agent_steps is not None else {}
        agent = Agent(chat_generator=self.generator, tools=tools, hooks=hooks, **steps)
        schema = (
            {"generation_kwargs": schema_kwargs(self.generator, response_format)}
            if response_format
            else {}
        )
        result = await run_attached(
            agent, to_haystack(messages), turn=turn.hooks(to_haystack) if turn else None, **schema
        )
        if turn is not None:
            await turn.gate()  # the run's gate hook judged it; raises Blocked on a refusal
        replies = result.get("messages") or []
        text = replies[-1].text if replies else None
        if not response_format:
            return text
        if not text:
            # no answer to validate
            logger.warning(
                "No {} in the reply (finish_reason={})",
                response_format.__name__,
                replies[-1].meta.get("finish_reason") if replies else None,
            )
            return None
        try:
            return response_format.model_validate_json(text)
        except ValidationError:
            if turn is not None and turn.guard is not None and turn.guard.finished is not None:
                # The finish replaced the JSON: refuse rather than return the original.
                raise turn.unparsable() from None
            raise


__all__ = ["Run", "get_user_message", "get_user_text"]
