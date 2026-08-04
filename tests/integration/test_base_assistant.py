"""
Integration tests for BaseAssistant.
"""

import pytest
import pytest_asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from django_ai_sdk import Assistant
from django_ai_sdk.common import ChatMessage
from django_ai_sdk.storage.memory import MemoryStorageAdapter, MemoryStore
from django_ai_sdk.protocols.vercel import VercelProtocolHandler
from tests.factories.schemas import ChatMessageFactory


@pytest.mark.django_db
class TestBaseAssistant:
    """Test suite for BaseAssistant core functionality."""

    @pytest_asyncio.fixture
    async def assistant(self):
        """Create a test assistant with memory storage."""

        class TestAssistant(Assistant):
            name = "test_assistant"
            model = "gpt-4o-mini"
            instructions = ["You are a test assistant"]
            protocol = VercelProtocolHandler  # Class, not instance!
            storage_adapter = MemoryStorageAdapter

            async def get_pipeline_adapter(self, thread_id: str | None = None):
                return MagicMock()

        return TestAssistant()

    @pytest_asyncio.fixture
    async def thread_id(self):
        """Create a test thread ID - must be valid UUID for Django ORM."""
        return str(uuid.uuid4())

    @pytest.mark.asyncio
    async def test_assistant_creates_thread(self, assistant, thread_id):
        """Test that assistant properly creates threads."""
        # Create thread FIRST
        await MemoryStorageAdapter.create_thread(
            title="Test Thread",
            metadata={"assistant_id": "test_assistant"},
            thread_id=thread_id
        )

        # Now get storage adapter for existing thread
        storage = await assistant.get_storage_adapter(thread_id)

        # Thread should be accessible
        assert storage is not None
        assert storage.thread_id == thread_id

    @pytest.mark.asyncio
    async def test_message_storage_flow(self, assistant, thread_id):
        """Test complete message storage flow."""
        # Create thread FIRST
        await MemoryStorageAdapter.create_thread(
            title="Test Thread",
            metadata={"assistant_id": "test_assistant"},
            thread_id=thread_id
        )

        # Now get storage adapter
        storage = await assistant.get_storage_adapter(thread_id)
        assert storage is not None

        # Store user message
        user_msg = ChatMessageFactory.build(role="user")
        message_id = await storage.store_chat_message(user_msg)

        # Verify storage
        history = await storage.get_messages()
        assert len(history) == 1
        assert history[0].content == user_msg.content
        assert history[0].role == "user"

    @pytest.mark.asyncio
    async def test_message_rating_flow(self, assistant, thread_id):
        """Test complete message rating workflow."""
        # Create thread FIRST
        await MemoryStorageAdapter.create_thread(
            title="Test Thread",
            metadata={"assistant_id": "test_assistant"},
            thread_id=thread_id
        )

        storage = await assistant.get_storage_adapter(thread_id)
        assert storage is not None

        # Store message
        msg = ChatMessageFactory.build()
        message_id = await storage.store_chat_message(msg)

        # Rate the message
        success = await storage.rate_message(message_id, rating=1)
        assert success is True

        # Verify rating persisted by checking MemoryStore directly
        stored_messages = MemoryStore.get_messages(thread_id)
        assert len(stored_messages[0].feedbacks) == 1
        assert stored_messages[0].feedbacks[0]["rating"] == 1

    @pytest.mark.asyncio
    async def test_conversation_history_retrieval(self, assistant, thread_id):
        """Test conversation history retrieval."""
        # Create thread FIRST
        await MemoryStorageAdapter.create_thread(
            title="Test Thread",
            metadata={"assistant_id": "test_assistant"},
            thread_id=thread_id
        )

        storage = await assistant.get_storage_adapter(thread_id)
        assert storage is not None

        # Create conversation history
        messages = [
            ChatMessageFactory.build(role="user", content="Question 1"),
            ChatMessageFactory.build(role="assistant", content="Answer 1"),
            ChatMessageFactory.build(role="user", content="Question 2"),
        ]

        for msg in messages:
            await storage.store_chat_message(msg)

        # Retrieve and verify
        history = await storage.get_messages()
        assert len(history) == 3
        assert history[0].role == "user"
        assert history[1].role == "assistant"
        assert history[2].role == "user"

    @pytest.mark.asyncio
    async def test_thread_not_found_error(self, assistant):
        """Test that accessing non-existent thread raises error."""
        # Use a valid UUID format but one that doesn't exist
        nonexistent_id = str(uuid.uuid4())
        storage = MemoryStorageAdapter(thread_id=nonexistent_id)

        # Should raise ValueError when trying to store without creating thread first
        msg = ChatMessageFactory.build()
        with pytest.raises(ValueError, match="not found in memory store"):
            await storage.store_chat_message(msg)

    @pytest.mark.asyncio
    async def test_protocol_message_conversion(self, assistant):
        """Test protocol handler converts messages correctly."""
        protocol = assistant.protocol_handler  # Use protocol_handler, not protocol

        # Create mock Vercel message objects with .parts attribute
        class MockPart:
            def __init__(self, type, text) -> None:
                self.type = type
                self.text = text

        class MockMessage:
            def __init__(self, role, text) -> None:
                self.role = role
                self.parts = [MockPart("text", text)]

        # Test converting from protocol format
        protocol_messages = [
            MockMessage("user", "Hello"),
            MockMessage("assistant", "Hi there!"),
        ]

        chat_messages = protocol.to_chat_messages(protocol_messages)

        assert len(chat_messages) == 2
        assert chat_messages[0].role == "user"
        assert chat_messages[1].role == "assistant"


    @pytest.mark.asyncio
    async def test_get_storage_adapter_falls_back_for_unknown_thread(self, assistant):
        """get_storage_adapter() must return configured adapter for unknown threads."""
        unknown_thread_id = str(uuid.uuid4())
        # No thread created in any adapter — should fall back to storage_adapter
        storage = await assistant.get_storage_adapter(unknown_thread_id)
        assert storage is not None
        assert storage.thread_id == unknown_thread_id

    @pytest.mark.asyncio
    async def test_get_storage_adapter_returns_none_for_none_thread_id(self, assistant):
        """get_storage_adapter() must return None when thread_id is None."""
        storage = await assistant.get_storage_adapter(None)
        assert storage is None

    @pytest.mark.asyncio
    async def test_stream_rejects_non_pipeline(self):
        """Stream.__init__ must raise TypeError if given a non-Pipeline object."""
        from django_ai_sdk.adapters.base import Stream
        from unittest.mock import MagicMock

        with pytest.raises(TypeError, match="Pipeline"):
            Stream(pipeline="not a pipeline", generator=MagicMock())


@pytest.mark.django_db
class TestStreamWriterIntegration:
    """Test StreamWriter integration with BaseAssistant."""

    @pytest_asyncio.fixture
    async def assistant_with_storage(self):
        """Create assistant with storage enabled."""

        class TestAssistant(Assistant):
            name = "test_assistant"
            model = "gpt-4o-mini"
            instructions = ["You are a test assistant"]
            protocol = VercelProtocolHandler
            storage_adapter = MemoryStorageAdapter

            async def get_pipeline_adapter(self, thread_id: str | None = None, *args, **kwargs):
                return MagicMock()

        return TestAssistant()

    @pytest.mark.asyncio
    async def test_stream_writer_creates_message_with_id(self, assistant_with_storage):
        """Test that StreamWriter creates message with proper ID."""
        from django_ai_sdk.common import StreamWriter

        message_id = str(uuid.uuid4())
        stream_writer = StreamWriter(
                        message_id=message_id,
            model="test-model",
            role="assistant",
            storage_callback=None,
        )

        # Verify ID is set
        assert stream_writer.message.id == message_id
        assert isinstance(uuid.UUID(message_id), uuid.UUID)

    @pytest.mark.asyncio
    async def test_stream_writer_requires_id(self):
        """Test that StreamWriter requires message_id parameter."""
        from django_ai_sdk.common import StreamWriter

        # Should work with valid ID
        stream_writer = StreamWriter(
                        message_id=str(uuid.uuid4()),
            model="test-model",
            role="assistant",
        )
        assert stream_writer.message.id is not None


@pytest.mark.django_db
class TestTitleGenerationPrompt:
    """`get_title_generation_prompt` is overridable and defaults to a
    length-capped prompt, and `generate_thread_title` must honour whatever
    the assistant returns from it."""

    @pytest_asyncio.fixture
    async def assistant(self):
        class TestAssistant(Assistant):
            name = "test_assistant"
            model = "gpt-4o-mini"
            instructions = ["You are a test assistant"]
            protocol = VercelProtocolHandler
            storage_adapter = MemoryStorageAdapter

            async def get_pipeline_adapter(self, thread_id: str | None = None, *args, **kwargs):
                return MagicMock()

        return TestAssistant()

    def test_default_prompt_mentions_column_max_length(self, assistant):
        from django_ai_sdk.conversation.models import Thread

        max_length = Thread._meta.get_field("title").max_length
        prompt = assistant.get_title_generation_prompt()

        assert str(max_length) in prompt

    @pytest.mark.asyncio
    async def test_generate_thread_title_uses_overridden_prompt(self, assistant):
        from django_ai_sdk.common import Prompt
        from django_ai_sdk.conversation.utils import generate_thread_title

        custom_prompt = Prompt("Custom title prompt, stay under 10 characters.")
        assistant.get_title_generation_prompt = MagicMock(return_value=custom_prompt)
        assistant.run = AsyncMock(return_value="Result")

        await generate_thread_title(
            assistant=assistant,
            messages=[ChatMessage(role="user", content="hi")],
        )

        assert assistant.run.call_args.kwargs["system_prompt"] == custom_prompt
