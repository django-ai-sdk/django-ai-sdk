"""
Factory for ChatMessage objects.
"""

import uuid
from factory.base import Factory
from factory.declarations import LazyAttribute, Trait
from factory.faker import Faker
from django_ai_sdk.common import ChatMessage


class ChatMessageFactory(Factory):
    """Factory for creating ChatMessage objects for testing."""

    class Meta:
        model = ChatMessage

    id = LazyAttribute(lambda _: str(uuid.uuid4()))
    role = "assistant"
    content = Faker("paragraph", nb_sentences=3)
    adapter_type = "openai"
    model = "gpt-4o-mini"

    # Optional fields with defaults
    tool_calls = []
    sources = []
    reasoning = None
    finish_reason = "stop"

    class Params:
        """Factory parameters for creating different message types."""

        user = Trait(
            role="user",
            adapter_type="",
            model="",
            content=Faker("sentence", nb_words=6),
        )

        assistant = Trait(role="assistant", content=Faker("paragraph"))

        system = Trait(
            role="system", content=Faker("sentence", nb_words=10)
        )
