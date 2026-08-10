"""
Integration tests for Storage Adapters.
"""

import pytest
import pytest_asyncio
import uuid

from django_ai_sdk.common import ChatMessage
from django_ai_sdk.storage.memory import MemoryStorageAdapter, MemoryStore
from django_ai_sdk.storage.base import BaseStorageAdapter
from tests.factories.schemas import ChatMessageFactory


class TestMemoryStorageAdapter:
    """Test suite for MemoryStorageAdapter."""

    @pytest_asyncio.fixture
    async def storage(self):
        """Create memory storage adapter."""
        thread_id = str(uuid.uuid4())  # Use valid UUID
        return MemoryStorageAdapter(thread_id=thread_id)

    @pytest_asyncio.fixture
    async def thread_context(self, storage):
        """Create thread context for storage."""
        MemoryStore.create_thread(
            storage.thread_id,
            title="Test Thread",
            agent_id="test_agent",
            model="gpt-4o-mini",
        )
        return storage

    @pytest.mark.asyncio
    async def test_requires_explicit_thread_creation(self, storage):
        """Test that memory storage requires explicit thread creation."""
        msg = ChatMessageFactory.build()

        with pytest.raises(ValueError, match="not found in memory store"):
            await storage.store_chat_message(msg)

    @pytest.mark.asyncio
    async def test_store_and_retrieve_message(self, thread_context):
        """Test storing and retrieving a message."""
        msg = ChatMessageFactory.build(assistant=True)

        message_id = await thread_context.store_chat_message(msg)
        assert message_id is not None

        history = await thread_context.get_messages()
        assert len(history) == 1
        assert history[0].content == msg.content

    @pytest.mark.asyncio
    async def test_rating_persists_correctly(self, thread_context):
        """Test that message rating persists correctly."""
        msg = ChatMessageFactory.build()
        message_id = await thread_context.store_chat_message(msg)

        # Rate as good
        success = await thread_context.rate_message(message_id, rating=1)
        assert success is True

        # Verify rating persisted by checking MemoryStore directly
        stored_messages = MemoryStore.get_messages(thread_context.thread_id)
        assert len(stored_messages[0].feedbacks) == 1
        assert stored_messages[0].feedbacks[0]["rating"] == 1

    @pytest.mark.asyncio
    async def test_negative_rating(self, thread_context):
        """Test negative rating functionality."""
        msg = ChatMessageFactory.build()
        message_id = await thread_context.store_chat_message(msg)

        # Rate as bad
        success = await thread_context.rate_message(message_id, rating=-1)
        assert success is True

        # Verify rating persisted by checking MemoryStore directly
        stored_messages = MemoryStore.get_messages(thread_context.thread_id)
        assert len(stored_messages[0].feedbacks) == 1
        assert stored_messages[0].feedbacks[0]["rating"] == -1

    @pytest.mark.asyncio
    async def test_rating_nonexistent_message(self, thread_context):
        """Test rating a message that doesn't exist."""
        success = await thread_context.rate_message("nonexistent_id", rating=1)
        assert success is False

    @pytest.mark.asyncio
    async def test_conversation_history_ordering(self, thread_context):
        """Test that conversation history is returned in correct order."""
        # Store multiple messages
        messages = [
            ChatMessageFactory.build(role="user", content="First"),
            ChatMessageFactory.build(role="assistant", content="Response 1"),
            ChatMessageFactory.build(role="user", content="Second"),
        ]

        for msg in messages:
            await thread_context.store_chat_message(msg)

        history = await thread_context.get_messages()
        assert len(history) == 3
        assert history[0].role == "user"
        assert history[0].content == "First"
        assert history[1].role == "assistant"
        assert history[2].role == "user"

    @pytest.mark.asyncio
    async def test_delete_message(self, thread_context):
        """Test message deletion (soft delete)."""
        msg = ChatMessageFactory.build()
        message_id = await thread_context.store_chat_message(msg)

        # Delete message
        success = await thread_context.delete_message(message_id)
        assert success is True

        # Message should not appear in history
        history = await thread_context.get_messages()
        assert len(history) == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent_message(self, thread_context):
        """Test deleting a message that doesn't exist."""
        success = await thread_context.delete_message("nonexistent_id")
        assert success is False

    @pytest.mark.asyncio
    async def test_thread_metadata_preserved(self, thread_context):
        """Test that thread metadata is preserved."""
        thread = MemoryStore.get_thread(thread_context.thread_id)

        assert thread is not None
        assert thread.title == "Test Thread"
        assert thread.agent_id == "test_agent"
        assert thread.model == "gpt-4o-mini"


class TestMemoryStore:
    """Test suite for MemoryStore singleton."""

    @pytest.mark.asyncio
    async def test_thread_creation_with_metadata(self):
        """Test thread creation with full metadata."""
        thread_id = str(uuid.uuid4())

        thread = MemoryStore.create_thread(
            thread_id,
            title="My Thread",
            agent_id="my_agent",
            model="gpt-4",
            metadata={"custom": "value"},
        )

        assert thread.id == thread_id
        assert thread.title == "My Thread"
        assert thread.metadata == {"custom": "value"}

    @pytest.mark.asyncio
    async def test_list_threads(self):
        """Test listing all threads."""
        # Create a few threads
        thread_ids = []
        for i in range(3):
            tid = str(uuid.uuid4())
            MemoryStore.create_thread(tid, title=f"Thread {i}")
            thread_ids.append(tid)

        # List threads
        threads = MemoryStore.list_threads()

        # Verify our threads are in the list
        thread_ids_found = [t.id for t in threads]
        for tid in thread_ids:
            assert tid in thread_ids_found

    @pytest.mark.asyncio
    async def test_list_threads_by_user(self):
        """Test filtering threads by user."""
        user_id = "user_123"

        # Create threads for user
        tid1 = str(uuid.uuid4())
        tid2 = str(uuid.uuid4())
        MemoryStore.create_thread(tid1, title="User Thread 1", user_id=user_id)
        MemoryStore.create_thread(tid2, title="User Thread 2", user_id=user_id)

        # Create thread for different user
        tid3 = str(uuid.uuid4())
        MemoryStore.create_thread(tid3, title="Other Thread", user_id="other_user")

        # List threads for user
        threads = MemoryStore.list_threads(user_id=user_id)

        assert len(threads) == 2
        for t in threads:
            assert t.user_id == user_id


class TestStorageAdapterInterface:
    """Test suite for BaseStorageAdapter interface compliance."""

    @pytest.mark.asyncio
    async def test_storage_adapter_interface_methods(self):
        """Test that storage adapter implements all required methods."""
        adapter = MemoryStorageAdapter(thread_id="test")

        # Check all required methods exist
        assert hasattr(adapter, "store_chat_message")
        assert hasattr(adapter, "get_messages")
        assert hasattr(adapter, "rate_message")
        assert hasattr(adapter, "delete_message")
        assert hasattr(adapter, "storage_callback")
