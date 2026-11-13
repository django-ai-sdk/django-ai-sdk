"""The Run adapter: one-off model calls and headless tool loops, as Haystack Agent runs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage as HaystackChatMessage
from haystack.dataclasses import ImageContent

from django_ai_sdk.generators import schema_kwargs
from django_ai_sdk.logger import get_logger

if TYPE_CHECKING:
    from django_ai_sdk.adapters.interfaces import T
    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.oversight.plugin import TurnRequest

logger = get_logger(__name__)


def get_user_text(message: ChatMessage) -> str:
    """User text followed by the context line of each attachment."""
    hints = [a.context for a in message.attachments if a.context]
    return "\n\n".join([message.content, *hints]) if hints else message.content


def user_content(message: ChatMessage, text: str) -> str | None:
    """The content part of `text`, joined as `get_user_text` joins it. None if it no longer splits."""
    hints = [a for a in message.attachments if a.context]
    parts = text.rsplit("\n\n", len(hints)) if hints else [text]
    return parts[0] if len(parts) == len(hints) + 1 else None


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
        request: TurnRequest | None = None,
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
        request: TurnRequest | None = None,
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
        request: TurnRequest | None = None,
    ) -> T | str | None:
        """Run the generator as a Haystack Agent: once, or as a tool loop with `tools`.

        `request` names the turn (see `oversight.plugin`); the run's oversight
        hooks join in `run_agent`. Raises `Blocked` when the turn's gate refused
        the question or its finish withheld the answer.
        """
        from django_ai_sdk.oversight.plugin import TurnRequest, run_agent

        request = request or TurnRequest()
        request.schema = response_format
        options: dict[str, Any] = {"hooks": hooks}
        if max_agent_steps is not None:
            options["max_agent_steps"] = max_agent_steps
        agent = Agent(chat_generator=self.generator, tools=tools, **options)
        converted = self.get_messages(messages)
        if system_prompt:
            # A message, not the Agent's system_prompt, which Haystack renders as a template.
            converted = [HaystackChatMessage.from_system(system_prompt), *converted]
        schema = (
            {"generation_kwargs": schema_kwargs(self.generator, response_format)}
            if response_format
            else {}
        )
        result = await run_agent(agent, converted, hook_context=request.context(), **schema)
        if request.turn is not None:
            request.turn.raise_if_blocked()
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
        return response_format.model_validate_json(text)


__all__ = ["Run", "get_user_message", "get_user_text", "user_content"]
