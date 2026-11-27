from datetime import datetime, timezone

from polyfactory.factories.pydantic_factory import ModelFactory

from django_ai_sdk.common import ChatMessage
from django_ai_sdk.views.schemas import Message, MessagePart
from django_ai_sdk.storage.schemas import ThreadInfo


def chat_message(
    role: str, text: str, message_id: str | None = None
) -> Message:
    return Message(
        role=role,
        parts=[MessagePart(type="text", text=text)],
        id=message_id,
    )


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ThreadInfoFactory(ModelFactory[ThreadInfo]):
    id = "thread-1"
    title = "Test Thread"
    agent_id = "test-agent"
    model = "gpt-4"
    user_id = None
    created_at = _now()
    updated_at = _now()
    message_count = 0

    __set_as_default_factory_for_type__ = False


class ChatMessageFactory(ModelFactory[ChatMessage]):
    role = "assistant"
    content = "Hello"
    model = "gpt-4o-mini"
    finish_reason = "stop"
    tool_calls = []
    sources = []
    attachments = []
    reasoning = None
    errors = []
    usage = {}
    metadata = {}

    __set_as_default_factory_for_type__ = False
