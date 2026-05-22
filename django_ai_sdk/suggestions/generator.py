from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel

from django_ai_sdk.common import ChatMessage, prompt
from django_ai_sdk.logger import get_logger

if TYPE_CHECKING:
    from django_ai_sdk.assistant import Assistant

logger = get_logger(__name__)


class SuggestionGenerator(Protocol):
    """Generate follow-up questions based on a response."""

    def __init__(self, assistant: Assistant) -> None: ...

    async def generate(self, messages: list[ChatMessage], response: str) -> list[str]:
        """Generate 2-3 follow-up suggestions.

        Args:
            messages: Full conversation history
            response: The assistant's response to generate suggestions from

        Returns:
            List of suggested follow-up questions (max 3)
        """
        ...


class FollowUpSuggestions(BaseModel):
    follow_ups: list[str]


def format_conversation(messages: list[ChatMessage], response: str) -> str:
    """Format conversation history into a string for the prompt."""
    lines = [
        f"{msg.role.upper()}: {msg.content}"
        for msg in messages
        if msg.role in ("user", "assistant") and msg.content
    ]
    lines.append(f"ASSISTANT: {response}")
    return "\n\n".join(lines)


class DefaultSuggestionGenerator:
    """Default implementation using LLM to generate contextual suggestions.

    Customizable via prompt parameter. Suggestions are disabled (returns empty list) if:
    - Assistant.get_suggestion_generator() is not overridden (returns None by default)
    """

    DEFAULT_PROMPT = prompt("""\
        You are a helpful assistant that suggests follow-up questions.
        Task: Suggest 2-3 relevant follow-up questions that the user might naturally ask next
        based on the conversation and the assistant's previous response.

        Guidelines:
        - Write questions from the user's point of view, as if they're asking the assistant.
        - Make questions concise, clear, and directly related to the discussed topic.
        - Suggest follow-ups that make sense given the context and don't repeat what was already covered.
        - Detect the conversation's language and use the same language for questions.
    """)

    def __init__(
        self,
        assistant: Assistant,
        prompt: str | None = None,
    ) -> None:
        self.assistant = assistant
        self.prompt = prompt or self.DEFAULT_PROMPT

    async def generate(self, messages: list[ChatMessage], response: str) -> list[str]:
        """Generate suggestions using LLM via the assistant.

        Returns empty list if:
        - response is empty
        - LLM call fails for any reason
        """
        if not response:
            return []

        try:
            conversation = format_conversation(messages, response)
            system_prompt = prompt(f"""\
                {self.prompt}

                Conversation:
                {conversation}
                Based on this conversation, suggest follow-up questions.
            """)

            result = await self.assistant.run(
                messages=messages,
                system_prompt=system_prompt,
                response_format=FollowUpSuggestions,
            )

            return result.follow_ups[:3] if result else []
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}", exc_info=True)

        return []
