from django_ai_sdk.logger import get_logger
from django_ai_sdk.storage.base import StorageAdapterRegistry
from django_ai_sdk.storage.schemas import ThreadInfo

logger = get_logger(__name__)


class ThreadService:
    """
    Service for thread operations across all storage adapters.
    """

    @staticmethod
    async def create_thread(
        assistant_id: str,
        title: str,
        metadata: dict | None = None,
        user_id: str | None = None,
        thread_id: str | None = None,
    ) -> str:
        """
        Create a new thread in the appropriate storage.

        Uses the assistant's configured storage adapter class.

        Args:
            assistant_id: ID of the assistant creating the thread
            title: Thread title (usually first user message)
            metadata: Additional metadata (will include assistant_id)
            user_id: Optional user ID for the thread owner
            thread_id: Optional custom thread ID

        Returns:
            Thread ID (UUID string)
        """
        # Import registry
        # TODO: check if we can move to top-level import, but it breaks due to circular imports
        from django_ai_sdk.assistants.registry import registry

        assistant = registry.get(assistant_id)
        if not assistant:
            raise ValueError(f"Assistant not found: {assistant_id}")

        # Get storage adapter class from assistant
        storage_class = assistant.storage_adapter
        if storage_class is None:
            from django_ai_sdk.storage.memory import MemoryStorageAdapter

            storage_class = MemoryStorageAdapter

        # Create thread using storage class method
        metadata = metadata or {}
        metadata["assistant_id"] = assistant_id

        thread_id = await storage_class.create_thread(
            title=title, metadata=metadata, user_id=user_id, thread_id=thread_id
        )

        logger.debug(f"Created thread {thread_id} for assistant {assistant_id}")
        return thread_id

    @staticmethod
    async def get_assistant(thread_id: str) -> ThreadInfo | None:
        """
        Find thread by querying storage adapters (fastest first).

        Returns thread metadata including assistant_id.

        Args:
            thread_id: Thread ID to look up

        Returns:
            ThreadInfo with assistant_id, or None if not found
        """
        # Query storage adapters in order of speed
        for adapter_class in StorageAdapterRegistry.get_all_adapters():
            thread = await adapter_class.get_thread(thread_id)
            if thread:
                logger.debug(
                    f"Found thread {thread_id} in {adapter_class.__name__}, assistant: {thread.assistant_id}"
                )
                return thread

        logger.debug(f"Thread not found: {thread_id}")
        return None

    @staticmethod
    async def threads(user_id: str | None = None) -> list[ThreadInfo]:
        """
        List all threads from all storage adapters.

        Args:
            user_id: Optional filter by user ID

        Returns:
            List of ThreadInfo from all storage types
        """
        all_threads = []

        for adapter_class in StorageAdapterRegistry.get_all_adapters():
            threads = await adapter_class.list_threads(user_id)
            all_threads.extend(threads)
            logger.debug(f"Found {len(threads)} threads in {adapter_class.__name__}")

        # Sort by updated_at (newest first)
        all_threads.sort(key=lambda t: t.updated_at, reverse=True)

        logger.debug(f"Total threads: {len(all_threads)}")
        return all_threads

    @staticmethod
    async def update_thread(
        thread_id: str, title: str | None = None, metadata: dict | None = None
    ) -> bool:
        """
        Update thread metadata.

        Finds thread in appropriate storage and updates it.

        Args:
            thread_id: Thread ID to update
            title: New title (optional)
            metadata: New metadata to merge (optional)

        Returns:
            True if updated, False if not found
        """
        for adapter_class in StorageAdapterRegistry.get_all_adapters():
            # Try to update in this storage
            success = await adapter_class.update_thread(thread_id, title, metadata)
            if success:
                logger.debug(f"Updated thread {thread_id} in {adapter_class.__name__}")
                return True

        return False

    @staticmethod
    async def delete_thread(thread_id: str) -> bool:
        """
        Delete a thread and all its messages.

        Args:
            thread_id: Thread ID to delete

        Returns:
            True if deleted, False if not found
        """
        for adapter_class in StorageAdapterRegistry.get_all_adapters():
            success = await adapter_class.delete_thread(thread_id)
            if success:
                logger.debug(f"Deleted thread {thread_id} from {adapter_class.__name__}")
                return True

        return False
