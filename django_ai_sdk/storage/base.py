from abc import ABC, abstractmethod
from enum import IntEnum
from typing import TYPE_CHECKING, Any, ClassVar

from django_ai_sdk.storage.schemas import ThreadInfo

if TYPE_CHECKING:
    from django_ai_sdk.common import ChatMessage


class StorageType(IntEnum):
    """
    Operation cost levels for storage adapters.

    Lower numbers = faster/cheaper operations
    Higher numbers = slower/more expensive operations

    Used by StorageAdapterRegistry to order operations
    """

    MEMORY = 1  # In-process, no I/O (fastest)
    FILE = 2  # Local disk I/O
    DATABASE = 3  # Network/database I/O
    REST_API = 4  # External service call (slowest)


class StorageAdapterRegistry:
    """
    Global registry for storage adapter classes.

    Tracks all available storage adapters.
    Used by ThreadService to efficiently query across storage types.

    Example:
        # Register adapters (done in each adapter module)
        StorageAdapterRegistry.register(MemoryStorageAdapter, StorageType.MEMORY)
        StorageAdapterRegistry.register(DbStorageAdapter, StorageType.DATABASE)

        # Get adapters sorted by speed
        for adapter in StorageAdapterRegistry.get_all_adapters():
            thread = await adapter.get_thread(thread_id)
    """

    _adapters: ClassVar[dict[type["BaseStorageAdapter"], StorageType]] = {}

    @classmethod
    def register(
        cls,
        adapter_class: type["BaseStorageAdapter"],
        intensiveness: StorageType,
    ) -> None:
        """
        Register a storage adapter class with its operation cost.

        Args:
            adapter_class: The storage adapter class to register
            intensiveness: Operation cost level (MEMORY, DATABASE, etc.)
        """
        cls._adapters[adapter_class] = intensiveness

    @classmethod
    def get_all_adapters(cls) -> list[type["BaseStorageAdapter"]]:
        """
        Get all registered adapters sorted by intensiveness (fastest first).

        Returns:
            List of adapter classes ordered from fastest to slowest
        """
        return sorted(cls._adapters.keys(), key=lambda a: cls._adapters[a])

    @classmethod
    def get_intensiveness(cls, adapter_class: type["BaseStorageAdapter"]) -> StorageType:
        """Get the intensiveness level for a specific adapter class."""
        return cls._adapters.get(adapter_class, StorageType.DATABASE)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered adapters. Useful for testing."""
        cls._adapters.clear()


class BaseStorageAdapter(ABC):
    """
    Base interface for conversation storage with ChatMessage callbacks.

    Storage adapters provide a clean interface for persisting conversations
    in different backends (database, memory, files, etc.).

    Two usage patterns:
    1. Class methods - for operations across all threads (create, list, find)
    2. Instance methods - for operations on a specific thread (history, messages)

    Example:
        # Class method - create thread (no instance needed)
        thread_id = await DbStorageAdapter.create_thread("Title", metadata)

        # Instance method - needs thread_id from constructor
        storage = DbStorageAdapter(thread_id)
        history = await storage.get_history()
    """

    def __init__(self, thread_id: str) -> None:
        """
        Initialize storage adapter bound to a specific thread.

        Args:
            thread_id: Unique identifier for the conversation thread
        """
        self.thread_id = thread_id

    # ============================================================================
    # CLASS METHODS - Thread Management
    # ============================================================================

    @classmethod
    @abstractmethod
    async def create_thread(
        cls,
        title: str,
        metadata: dict[str, Any] | None = None,
        user_id: str | None = None,
        thread_id: str | None = None,
    ) -> str:
        """
        Create a new thread in this storage.

        Args:
            title: Thread title
            metadata: Optional metadata dict (should include assistant_id)
            user_id: Optional user ID for the thread owner
            thread_id: Optional custom thread ID

        Returns:
            String thread_id (UUID) of the created thread
        """
        pass

    @classmethod
    @abstractmethod
    async def get_thread(cls, thread_id: str) -> ThreadInfo | None:
        """
        Get thread metadata by ID.

        Args:
            thread_id: Thread ID to look up

        Returns:
            ThreadInfo if found, None otherwise
        """
        pass

    @classmethod
    @abstractmethod
    async def list_threads(cls, user_id: str | None = None) -> list[ThreadInfo]:
        """
        List all threads in this storage.

        Args:
            user_id: Optional filter by user ID

        Returns:
            List of ThreadInfo objects
        """
        pass

    @classmethod
    @abstractmethod
    async def update_thread(
        cls,
        thread_id: str,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Update thread metadata.

        Args:
            thread_id: Thread ID to update
            title: New title (optional)
            metadata: New metadata dict (optional, merges with existing)

        Returns:
            True if updated successfully, False if thread not found
        """
        pass

    @classmethod
    @abstractmethod
    async def delete_thread(cls, thread_id: str) -> bool:
        """
        Delete a thread and all its messages.

        Args:
            thread_id: Thread ID to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @classmethod
    @abstractmethod
    async def delete_all_threads(cls) -> int:
        """
        Delete all threads and their messages.

        Returns:
            Number of threads deleted
        """
        pass

    # ============================================================================
    # INSTANCE METHODS - Thread-Specific Operations
    # ============================================================================

    @abstractmethod
    async def get_messages(self) -> list["ChatMessage"]:
        """
        Retrieve all ChatMessages for this thread, ordered by creation.

        Should exclude deleted messages.

        Returns:
            List of ChatMessage objects representing the conversation history
        """
        pass

    @abstractmethod
    async def store_chat_message(self, chat_message: "ChatMessage") -> str:
        """
        Store a ChatMessage to this thread.

        Args:
            chat_message: ChatMessage instance to store

        Returns:
            String identifier for the stored message
        """
        pass

    @abstractmethod
    async def storage_callback(self, chat_message: "ChatMessage") -> str | None:
        """
        Callback for StreamWriter to auto-store messages.

        Called automatically when streaming completes to persist
        the assistant's response.

        Args:
            chat_message: The ChatMessage to store

        Returns:
            Message ID string if stored, None if failed
        """
        pass

    @abstractmethod
    async def rate_message(self, message_id: str, rating: int) -> bool:
        """
        Rate a message in this thread.

        Args:
            message_id: Message ID to rate
            rating: 1 for good, -1 for bad

        Returns:
            True if rated successfully, False if message not found
        """
        pass

    @abstractmethod
    async def delete_message(self, message_id: str) -> bool:
        """
        Soft delete a message in this thread.

        Args:
            message_id: Message ID to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def restore_message(self, message_id: str) -> bool:
        """
        Restore a soft-deleted message in this thread.

        Args:
            message_id: Message ID to restore

        Returns:
            True if restored, False if not found
        """
        pass
