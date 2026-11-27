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

ATTACHMENT_CONTEXT_META_KEY = "django_ai_sdk.attachment_context_at"


def get_user_text(message: ChatMessage) -> str:
    hints = [a.context for a in message.attachments if a.context]
    return "\n\n".join([message.content, *hints]) if hints else message.content


def user_content(message: ChatMessage, text: str) -> str | None:
    hints = [a for a in message.attachments if a.context]
    parts = text.rsplit("\n\n", len(hints)) if hints else [text]
    return parts[0] if len(parts) == len(hints) + 1 else None


def get_user_message(message: ChatMessage) -> HaystackChatMessage:
    text = get_user_text(message)
    meta = {ATTACHMENT_CONTEXT_META_KEY: len(message.content)} if text != message.content else None
    images = [
        ImageContent(base64_image=a.data, mime_type=a.media_type)
        for a in message.attachments
        if a.data
    ]
    if not images:
        return HaystackChatMessage.from_user(text, meta=meta)
    return HaystackChatMessage.from_user(
        meta=meta, content_parts=[text, *images] if text else images
    )


class Run:
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
        conversation = [m for m in messages if m.role in ("user", "assistant")]
        converted: list[HaystackChatMessage] = []
        for msg in conversation:
            if msg.role == "user":
                converted.append(get_user_message(msg))
            elif msg.role == "assistant":
                converted.append(HaystackChatMessage.from_assistant(msg.content))
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
        """Raises `Blocked` when the gate blocked the question or finish withheld the answer."""
        from django_ai_sdk.oversight.plugin import TurnRequest, run_with_oversight

        request = request or TurnRequest()
        request.answer_schema = response_format
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
        result = await run_with_oversight(agent, converted, turn=request, **schema)
        request.raise_if_blocked()
        replies = result.get("messages") or []
        text = replies[-1].text if replies else None
        if not response_format:
            return text
        if not text:
            logger.warning(
                "No {} in the reply (finish_reason={})",
                response_format.__name__,
                replies[-1].meta.get("finish_reason") if replies else None,
            )
            return None
        return response_format.model_validate_json(text)


__all__ = ["Run", "get_user_message", "get_user_text", "user_content"]
