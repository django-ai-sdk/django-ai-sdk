import json
from typing import Protocol

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage as HaystackChatMessage
from haystack.utils import Secret

from django_ai_sdk.common import ChatMessage
from django_ai_sdk.logger import get_logger

logger = get_logger(__name__)


class SuggestionGenerator(Protocol):
    """Generate follow-up questions based on a response."""

    def generate(self, messages: list[ChatMessage], response: str) -> list[str]:
        """Generate 2-3 follow-up suggestions.

        Args:
            messages: Full conversation history
            response: The assistant's response to generate suggestions from

        Returns:
            List of suggested follow-up questions (max 3)
        """
        ...


class DefaultSuggestionGenerator:
    """Default implementation using LLM to generate contextual suggestions.

    Customizable via prompt parameter. Suggestions are disabled (returns empty list) if:
    - api_key is not provided (None)
    - Assistant.get_suggestion_generator() is not overridden (returns None by default)
    """

    DEFAULT_PROMPT = (
        "You are a helpful assistant that suggests follow-up questions.\n\n"
        "Task: Suggest 2-3 relevant follow-up questions that the user might naturally ask next, "
        "based on the conversation and the assistant's previous response.\n\n"
        "Guidelines:\n"
        "- Write questions from the user's point of view, as if they're asking the assistant.\n"
        "- Make questions concise, clear, and directly related to the discussed topic.\n"
        "- Suggest follow-ups that make sense given the context and don't repeat what was already covered.\n"
        "- Detect the conversation's language and use the same language for questions.\n"
        "- Return as a JSON object with a 'follow_ups' key containing an array of strings.\n\n"
        "Example output: {\"follow_ups\": [\"Question 1?\", \"Question 2?\", \"Question 3?\"]}"
    )

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base_url: str | None = None,
        prompt: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.prompt = prompt or self.DEFAULT_PROMPT

    def generate(self, messages: list[ChatMessage], response: str) -> list[str]:
        """Generate suggestions using an LLM based on conversation history.

        Returns empty list if:
        - response is empty
        - api_key is not configured
        - LLM call fails for any reason
        """
        if not response:
            return []

        if not self.api_key:
            return []

        try:
            generator = OpenAIChatGenerator(
                model=self.model,
                api_key=Secret.from_token(self.api_key),
                api_base_url=self.api_base_url,
            )

            # Format conversation history for context
            conversation_context = ""
            for msg in messages:
                role = getattr(msg, 'role', 'unknown').upper() if hasattr(msg, 'role') else 'unknown'
                content = getattr(msg, 'content', '')
                if content:
                    conversation_context += f"{role}: {content}\n\n"
            conversation_context += f"ASSISTANT: {response}"

            suggestion_messages = [
                HaystackChatMessage.from_system(self.prompt),
                HaystackChatMessage.from_user(
                    f"Conversation:\n{conversation_context}\n\nBased on this conversation, suggest follow-up questions."
                ),
            ]

            result = generator.run(messages=suggestion_messages)
            suggestions_text = result["replies"][0].text

            # Parse JSON response
            parsed = json.loads(suggestions_text)

            # Handle both formats: direct array or object with "follow_ups" key
            if isinstance(parsed, list):
                return parsed[:3]
            elif isinstance(parsed, dict) and "follow_ups" in parsed:
                follow_ups = parsed["follow_ups"]
                return follow_ups[:3] if isinstance(follow_ups, list) else []
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}", exc_info=True)

        return []
