from typing import Any

from asgiref.sync import async_to_sync

from django_ai_sdk.conversation.utils import generate_thread_title
from django_ai_sdk.logger import get_logger
from django_ai_sdk.storage.base import StorageAdapterRegistry
from django_ai_sdk.storage.schemas import ThreadDetail, ThreadInfo

logger = get_logger(__name__)


class ThreadService:
    """
    Service for thread operations across all storage adapters.

    All methods are async. Use the sync-prefixed aliases for sync contexts
    (e.g., DRF class-based views).
    """

    @staticmethod
    async def create_thread(
        assistant_id: str,
        messages: list[Any] | None = None,
        title: str = "",
        metadata: dict | None = None,
        user_id: str | None = None,
        thread_id: str | None = None,
    ) -> str:
        """
        Create a new thread in the appropriate storage.

        Uses the assistant's configured storage adapter class.

        Args:
            assistant_id: ID of the assistant creating the thread
            messages: Optional list of messages for auto-title generation
            title: Thread title. If empty and messages provided, auto-generated
            metadata: Additional metadata. Auto-populates model, assistant_name,
                     assistant_class, and created_via. Caller-provided values
                     take precedence over auto-generated ones.
            user_id: Optional user ID for the thread owner
            thread_id: Optional custom thread ID

        Returns:
            Thread ID (UUID string)
        """
        from django_ai_sdk.assistants.registry import registry

        assistant = registry.get(assistant_id)
        if not assistant:
            raise ValueError(f"Assistant not found: {assistant_id}")

        storage_class = assistant.storage_adapter
        if storage_class is None:
            from django_ai_sdk.storage.memory import MemoryStorageAdapter

            storage_class = MemoryStorageAdapter

        title = title or generate_thread_title(messages or [])

        metadata = metadata or {}
        auto_metadata = {
            "assistant_id": assistant_id,
            "model": assistant.model,
            "assistant_name": assistant.name or assistant.__class__.__name__,
            "assistant_class": assistant.__class__.__name__,
            "created_via": "create_thread",
        }
        auto_metadata.update(metadata)

        thread_id = await storage_class.create_thread(
            title=title, metadata=auto_metadata, user_id=user_id, thread_id=thread_id
        )

        logger.debug(f"Created thread {thread_id} for assistant {assistant_id}")
        return thread_id

    @staticmethod
    async def rate_message(thread_id: str, message_id: str, rating: int) -> bool:
        """
        Rate a message in a thread.

        Args:
            thread_id: Thread ID containing the message
            message_id: Message ID to rate
            rating: 1 for good, -1 for bad

        Returns:
            True if rated successfully

        Raises:
            ValueError: If thread or message not found
        """
        storage = await ThreadService.storage_for_thread(thread_id)
        success = await storage.rate_message(message_id, rating)
        if not success:
            raise ValueError("Message not found")
        return True

    @staticmethod
    async def delete_message(thread_id: str, message_id: str) -> bool:
        """
        Soft delete a message in a thread.

        Args:
            thread_id: Thread ID containing the message
            message_id: Message ID to delete

        Returns:
            True if deleted successfully

        Raises:
            ValueError: If thread or message not found
        """
        storage = await ThreadService.storage_for_thread(thread_id)
        success = await storage.delete_message(message_id)
        if not success:
            raise ValueError("Message not found")
        return True

    @staticmethod
    async def restore_message(thread_id: str, message_id: str) -> bool:
        """
        Restore a soft-deleted message in a thread.

        Args:
            thread_id: Thread ID containing the message
            message_id: Message ID to restore

        Returns:
            True if restored successfully

        Raises:
            ValueError: If thread or message not found
        """
        storage = await ThreadService.storage_for_thread(thread_id)
        success = await storage.restore_message(message_id)
        if not success:
            raise ValueError("Message not found")
        return True

    @staticmethod
    async def get_thread(thread_id: str) -> ThreadInfo | None:
        """
        Find thread by querying storage adapters.

        Returns thread metadata including assistant_id.

        Args:
            thread_id: Thread ID to look up

        Returns:
            ThreadInfo with assistant_id, or None if not found
        """
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

    @staticmethod
    async def delete_all_threads() -> int:
        """
        Delete all threads and their messages.

        Returns:
            Total number of threads deleted
        """
        total_deleted = 0
        for adapter_class in StorageAdapterRegistry.get_all_adapters():
            count = await adapter_class.delete_all_threads()
            if count and count > 0:
                logger.debug(f"Deleted {count} threads from {adapter_class.__name__}")
                total_deleted += count

        return total_deleted

    @staticmethod
    async def storage_for_thread(thread_id: str) -> Any:
        """Resolve a thread's storage adapter, instantiated and bound to the thread.

        Convenience for the common pattern: look up the thread, find its
        assistant, get the assistant's storage class, instantiate with
        thread_id. Returns a ready-to-use storage adapter instance.

        Raises ValueError if the thread does not exist or the assistant
        has no storage_adapter configured.
        """
        from django_ai_sdk.assistants.services import AssistantService  # noqa: PLC0415

        thread_info = await ThreadService.get_thread(thread_id)
        if thread_info is None:
            raise ValueError(f"Thread not found: {thread_id}")
        assistant = AssistantService.from_registry(thread_info.assistant_id)
        return ThreadService.get_storage(assistant)(thread_id)

    @staticmethod
    def get_storage(assistant: Any) -> type:
        """
        Get the storage adapter class for an assistant.

        Args:
            assistant: Assistant instance

        Returns:
            Storage adapter class (not instantiated)

        Raises:
            ValueError: If assistant has no storage_adapter configured
        """
        storage_class = assistant.storage_adapter
        if storage_class is None:
            raise ValueError(
                f"Assistant '{assistant.assistant_id}' has no storage_adapter configured"
            )
        return storage_class


# ============================================================================
# Thread history
# ============================================================================


async def aget_thread_history(thread_id: str) -> dict[str, Any]:
    """
    Get thread history: thread metadata and messages only.

    For memories, use GET /memories/thread/{thread_id}/
    For file metadata, use aget_thread_file_meta().

    Args:
        thread_id: Thread ID to retrieve history for

    Returns:
        Dict with thread and messages

    Raises:
        ValueError: If thread or assistant not found
    """
    from django_ai_sdk.assistants.services import AssistantService

    assistant = await AssistantService.get_assistant(thread_id)
    thread_detail: ThreadDetail = await assistant.history(thread_id)

    return {
        "thread": thread_detail.thread,
        "messages": thread_detail.messages,
    }


get_thread_history = async_to_sync(aget_thread_history)


# ============================================================================
# Thread file metadata
# ============================================================================


async def aget_thread_file_meta(thread_id: str) -> dict[str, Any]:
    """
    Get file metadata for a thread: file_count and file_memory_id.

    Does not raise if thread has no file memory — returns {file_count: 0, file_memory_id: None}.

    Args:
        thread_id: Thread ID to retrieve file metadata for

    Returns:
        Dict with file_count and file_memory_id

    Raises:
        ValueError: If thread not found in DB
    """
    from django_ai_sdk.conversation.models import Thread
    from django_ai_sdk.memories.models import Entry

    if not await Thread.objects.filter(id=thread_id).aexists():
        raise ValueError("Thread not found")

    thread = await Thread.objects.select_related("file_memory").aget(id=thread_id)
    file_memory_id = str(thread.file_memory_id) if thread.file_memory_id else None
    file_count = (
        await Entry.objects.filter(memory_id=thread.file_memory_id).acount()
        if file_memory_id
        else 0
    )

    return {
        "file_count": file_count,
        "file_memory_id": file_memory_id,
    }


get_thread_file_meta = async_to_sync(aget_thread_file_meta)


# ============================================================================
# Sync wrappers for use in sync contexts
# ============================================================================

create_thread = async_to_sync(ThreadService.create_thread)
rate_message = async_to_sync(ThreadService.rate_message)
delete_message = async_to_sync(ThreadService.delete_message)
restore_message = async_to_sync(ThreadService.restore_message)
get_thread = async_to_sync(ThreadService.get_thread)
threads = async_to_sync(ThreadService.threads)
update_thread = async_to_sync(ThreadService.update_thread)
delete_thread = async_to_sync(ThreadService.delete_thread)
delete_all_threads = async_to_sync(ThreadService.delete_all_threads)
