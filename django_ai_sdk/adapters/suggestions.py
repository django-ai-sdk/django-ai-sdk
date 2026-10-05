from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, Field

from django_ai_sdk.common import ChatMessage, prompt
from django_ai_sdk.logger import get_logger

if TYPE_CHECKING:
    from django_ai_sdk.agents.base import Agent

logger = get_logger(__name__)


class SuggestionGenerator(Protocol):
    """Generate follow-up questions based on a response."""

    def __init__(self, agent: Agent) -> None: ...

    async def generate(self, messages: list[ChatMessage], response: str) -> list[str]:
        """Generate 2-3 follow-up suggestions.

        Args:
            messages: Full conversation history
            response: The agent's response to generate suggestions from

        Returns:
            List of suggested follow-up questions (max 3)
        """
        ...


class FollowUpSuggestions(BaseModel):
    follow_ups: list[str] = Field(
        description="Messages the USER could send next to the agent, in the user's voice"
    )


def format_conversation(messages: list[ChatMessage], response: str) -> str:
    """Format conversation history into a string for the prompt."""
    # One label per side: the prompt speaks of the user and the agent.
    label = {"user": "USER", "assistant": "AGENT"}
    lines = [
        f"{label[msg.role]}: {msg.content}" for msg in messages if msg.role in label and msg.content
    ]
    lines.append(f"AGENT: {response}")
    return "\n\n".join(lines)


class DefaultSuggestionGenerator:
    """Default implementation using LLM to generate contextual suggestions.

    Customizable via prompt parameter. Suggestions are disabled (returns empty list) if:
    - Agent.get_suggestion_generator() is not overridden (returns None by default)
    """

    DEFAULT_PROMPT = prompt("""\
        You suggest what the user could say next in a conversation with an agent.
        Task: Write 2-3 short messages the USER might naturally send next, based on the
        conversation and the agent's last response. They are clickable suggestions: clicking
        one sends it to the agent as the user's own message.

        Guidelines:
        - Each suggestion is a message the user sends to the agent: a question or request
          to it. Never a question the agent asks the user ("Would you like me to…", "Do you have…",
          "Which … are you interested in?").
        - Make questions concise, clear, and directly related to the discussed topic.
        - Suggest follow-ups that make sense given the context and don't repeat what was already covered.
        - Detect the conversation's language and use the same language for questions.
    """)

    def __init__(
        self,
        agent: Agent,
        prompt: str | None = None,
    ) -> None:
        self.agent = agent
        self.prompt = prompt or self.DEFAULT_PROMPT

    async def generate(self, messages: list[ChatMessage], response: str) -> list[str]:
        """Generate suggestions using LLM via the agent.

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
                Based on this conversation, write the user's possible next messages.
            """)

            result = await self.agent.run(
                # The conversation is in the system prompt. Passed as chat messages
                # too, the model carries on as the assistant and suggests what *it*
                # would ask the user ("...the projects you'd like me to analyze?").
                messages=[ChatMessage(role="user", content="Suggest the follow-up questions.")],
                system_prompt=system_prompt,
                response_format=FollowUpSuggestions,
            )

            return result.follow_ups[:3] if isinstance(result, FollowUpSuggestions) else []
        except Exception as e:
            # Don't f-string `e` in: its text can contain braces (e.g. malformed
            # JSON from the model), which loguru's internal str.format() then
            # chokes on, turning this already-handled failure into a crash.
            logger.opt(exception=e).error("Error generating suggestions: {}", e)

        return []
