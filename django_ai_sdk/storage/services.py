from __future__ import annotations

from typing import TYPE_CHECKING, Any

from asgiref.sync import async_to_sync

from django_ai_sdk.assistants.services import AssistantService
from django_ai_sdk.logger import get_logger
from django_ai_sdk.permissions import (
    BasePermission,
    Operation,
    PermissionDenied,
    check_object_permissions,
    check_permissions,
    get_default_permissions,
)
from django_ai_sdk.storage.base import StorageAdapterRegistry

if TYPE_CHECKING:
    from django.contrib.auth.models import AbstractUser

    from django_ai_sdk.storage.schemas import ThreadDetail, ThreadInfo

logger = get_logger(__name__)


async def _get_thread(thread_id: str) -> ThreadInfo | None:
    """Look up a thread across all storage adapters"""
    for adapter_class in StorageAdapterRegistry.get_all_adapters():
        thread = await adapter_class.get_thread(thread_id)
        if thread:
            return thread
    return None


async def _get_storage(thread_info: ThreadInfo) -> Any:
    """Get a storage adapter instance for a thread without permission checks."""
    assistant = AssistantService.from_registry(thread_info.assistant_id)
    storage_class = ThreadService.get_storage(assistant)
    return storage_class(thread_info.id)


async def _check_object_permission(
    user: AbstractUser | None,
    operation: Operation,
    obj: ThreadInfo,
) -> None:
    """Object-level permission using the thread's owning assistant's permission classes."""
    try:
        assistant = AssistantService.from_registry(obj.assistant_id)
    except ValueError:
        assistant = None

    perm_classes: list[type[BasePermission]] = getattr(
        assistant, "permissions", get_default_permissions()
    )
    await check_permissions(user, operation, perm_classes)
    await check_object_permissions(user, operation, obj, perm_classes)


async def _check_permission(
    user: AbstractUser | None,
    operation: Operation,
    assistant: Any,
) -> None:
    """Check a global permission for an assistant."""
    perm_classes: list[type[BasePermission]] = getattr(
        assistant, "permissions", get_default_permissions()
    )
    await check_permissions(user, operation, perm_classes)


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
        *,
        user: AbstractUser | None,
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
            user: User for permission checking and thread ownership
            thread_id: Optional custom thread ID

        Returns:
            Thread ID (UUID string)

        Raises:
            PermissionDenied: If user has no CREATE_THREAD permission
        """
        from django_ai_sdk.assistants.registry import registry

        assistant = registry.get(assistant_id)
        if not assistant:
            raise ValueError(f"Assistant not found: {assistant_id}")

        await _check_permission(user, Operation.CREATE_THREAD, assistant)

        storage_class = assistant.storage_adapter
        title = title or ""
        default_metadata = {
            "assistant_id": assistant_id,
            "model": assistant.model,
            "assistant_name": assistant.name or assistant.__class__.__name__,
            "assistant_class": assistant.__class__.__name__,
            "created_via": "create_thread",
        }
        default_metadata.update(metadata or {})

        thread_id = await storage_class.create_thread(
            title=title,
            metadata=default_metadata,
            user=user,
            thread_id=thread_id,
        )

        logger.debug(f"Created thread {thread_id} for assistant {assistant_id}")
        return thread_id

    @staticmethod
    async def get_thread(thread_id: str, *, user: AbstractUser | None) -> ThreadInfo | None:
        """
        Find thread by querying storage adapters.

        Returns thread metadata including assistant_id.

        Args:
            thread_id: Thread ID to look up
            user: Required user for permission checking

        Returns:
            ThreadInfo with assistant_id, or None if not found

        Raises:
            PermissionDenied: If user has no VIEW_THREAD permission for the thread
        """
        thread = await _get_thread(thread_id)
        if thread:
            await _check_object_permission(user, Operation.VIEW_THREAD, thread)
            logger.debug(f"Found thread {thread_id} in adapter, assistant: {thread.assistant_id}")

        if not thread:
            logger.debug(f"Thread not found: {thread_id}")

        return thread

    @staticmethod
    async def threads(
        user: AbstractUser | None = None,
    ) -> list[ThreadInfo]:
        """
        List all threads from all storage adapters.

        Args:
            user: Optional user for filtering thread ownership

        Returns:
            List of ThreadInfo from all storage types
        """
        all_threads: list[ThreadInfo] = []

        for adapter_class in StorageAdapterRegistry.get_all_adapters():
            threads = await adapter_class.list_threads(user)
            for thread in threads:
                try:
                    await _check_object_permission(user, Operation.LIST_THREADS, thread)
                    all_threads.append(thread)
                except PermissionDenied:
                    continue
            logger.debug(f"Found {len(threads)} threads in {adapter_class.__name__}")

        all_threads.sort(key=lambda t: t.updated_at, reverse=True)

        logger.debug(f"Total threads: {len(all_threads)}")
        return all_threads

    @staticmethod
    async def update_thread(
        thread_id: str,
        *,
        user: AbstractUser | None,
        title: str | None = None,
        metadata: dict | None = None,
    ) -> bool:
        """
        Update thread metadata.

        Finds thread in appropriate storage and updates it.

        Args:
            thread_id: Thread ID to update
            user: Required user for permission checking
            title: New title (optional)
            metadata: New metadata to merge (optional)

        Returns:
            True if updated, False if not found

        Raises:
            PermissionDenied: If user has no UPDATE_THREAD permission
        """
        thread = await _get_thread(thread_id)
        if thread:
            await _check_object_permission(user, Operation.UPDATE_THREAD, thread)

        for adapter_class in StorageAdapterRegistry.get_all_adapters():
            success = await adapter_class.update_thread(thread_id, title, metadata)
            if success:
                logger.debug(f"Updated thread {thread_id} in {adapter_class.__name__}")
                return True

        return False

    @staticmethod
    async def delete_thread(thread_id: str, *, user: AbstractUser | None) -> bool:
        """
        Delete a thread and all its messages.

        Args:
            thread_id: Thread ID to delete
            user: Required user for permission checking

        Returns:
            True if deleted, False if not found

        Raises:
            PermissionDenied: If user has no DELETE_THREAD permission
        """
        thread = await _get_thread(thread_id)
        if thread:
            await _check_object_permission(user, Operation.DELETE_THREAD, thread)

        for adapter_class in StorageAdapterRegistry.get_all_adapters():
            success = await adapter_class.delete_thread(thread_id)
            if success:
                logger.debug(f"Deleted thread {thread_id} from {adapter_class.__name__}")
                return True

        return False

    @staticmethod
    async def delete_all_threads(*, user: AbstractUser | None) -> int:
        """
        Delete current user's threads and their messages.

        Args:
            user: Required user for permission checking and thread ownership

        Returns:
            Total number of threads deleted

        Raises:
            PermissionDenied: If user has no DELETE_ALL_THREADS permission
        """
        total_deleted = 0
        for adapter_class in StorageAdapterRegistry.get_all_adapters():
            threads = await adapter_class.list_threads(user)
            for thread in threads:
                try:
                    await _check_object_permission(user, Operation.DELETE_ALL_THREADS, thread)
                    if await adapter_class.delete_thread(thread.id):
                        total_deleted += 1
                except PermissionDenied:
                    continue
            logger.debug(f"Deleted threads from {adapter_class.__name__}")

        return total_deleted

    @staticmethod
    async def rate_message(
        thread_id: str,
        message_id: str,
        rating: int | None,
        feedback: str = "",
        *,
        user: AbstractUser | None,
    ) -> bool:
        """
        Rate a message in a thread.

        Args:
            thread_id: Thread ID containing the message
            message_id: Message ID to rate
            rating: 1 for good, -1 for bad
            user: Required user for permission checking

        Returns:
            True if rated successfully

        Raises:
            ValueError: If thread or message not found
            PermissionDenied: If user has no RATE_MESSAGE permission
        """
        thread = await _get_thread(thread_id)
        if not thread:
            raise ValueError("Thread not found")
        await _check_object_permission(user, Operation.RATE_MESSAGE, thread)

        storage = await _get_storage(thread)
        success = await storage.rate_message(message_id, rating, feedback, user=user)
        if not success:
            raise ValueError("Message not found")
        return True

    @staticmethod
    async def delete_message(thread_id: str, message_id: str, *, user: AbstractUser | None) -> bool:
        """
        Soft delete a message in a thread.

        Args:
            thread_id: Thread ID containing the message
            message_id: Message ID to delete
            user: Required user for permission checking

        Returns:
            True if deleted successfully

        Raises:
            ValueError: If thread or message not found
            PermissionDenied: If user has no DELETE_MESSAGE permission
        """
        thread = await _get_thread(thread_id)
        if not thread:
            raise ValueError("Thread not found")
        await _check_object_permission(user, Operation.DELETE_MESSAGE, thread)

        storage = await _get_storage(thread)
        success = await storage.delete_message(message_id)
        if not success:
            raise ValueError("Message not found")
        return True

    @staticmethod
    async def restore_message(
        thread_id: str, message_id: str, *, user: AbstractUser | None
    ) -> bool:
        """
        Restore a soft-deleted message in a thread.

        Args:
            thread_id: Thread ID containing the message
            message_id: Message ID to restore
            user: Required user for permission checking

        Returns:
            True if restored successfully

        Raises:
            ValueError: If thread or message not found
            PermissionDenied: If user has no RESTORE_MESSAGE permission
        """
        thread = await _get_thread(thread_id)
        if not thread:
            raise ValueError("Thread not found")
        await _check_object_permission(user, Operation.RESTORE_MESSAGE, thread)

        storage = await _get_storage(thread)
        success = await storage.restore_message(message_id)
        if not success:
            raise ValueError("Message not found")
        return True

    @staticmethod
    async def storage_for_thread(thread_id: str, *, user: AbstractUser | None = None) -> Any:
        """
        Resolve a thread's storage adapter, instantiated and bound to the thread.

        Convenience for the common pattern: look up the thread, find its
        assistant, get the assistant's storage class, instantiate with
        thread_id. Returns a ready-to-use storage adapter instance.

        Args:
            thread_id: Thread ID to get storage for
            user: User for permission checking

        Returns:
            Storage adapter instance bound to the thread

        Raises:
            ValueError: If the thread does not exist or has no storage
            PermissionDenied: If user has no VIEW_THREAD permission
        """
        thread = await _get_thread(thread_id)
        if thread is None:
            raise ValueError(f"Thread not found: {thread_id}")
        await _check_object_permission(user, Operation.VIEW_THREAD, thread)
        return await _get_storage(thread)

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


async def aget_thread_history(thread_id: str, user: AbstractUser | None = None) -> dict[str, Any]:
    """
    Get thread history: thread metadata and messages with feedbacks.

    Messages include feedbacks via the protocol handler which reads them from
    ChatMessage metadata. No feedbacks are duplicated in the response.

    For memories, use GET /memories/thread/{thread_id}/
    For file metadata, use aget_thread_file_meta().

    Args:
        thread_id: Thread ID to retrieve history for
        user: Optional user for permission checking

    Returns:
        Dict with thread and messages (each message includes feedbacks)

    Raises:
        ValueError: If thread or assistant not found
        PermissionDenied: If user has no VIEW_THREAD permission
    """
    from django_ai_sdk.assistants.services import AssistantService

    assistant = await AssistantService.get_assistant(thread_id, user=user)
    thread_detail: ThreadDetail = await assistant.history(thread_id, user=user)

    return {
        "thread": thread_detail.thread,
        "messages": thread_detail.messages,
    }


get_thread_history = async_to_sync(aget_thread_history)


# ============================================================================
# Thread file metadata
# ============================================================================


async def aget_thread_file_meta(thread_id: str, *, user: AbstractUser | None) -> dict[str, Any]:
    """
    Get file metadata for a thread: file_count and file_memory_id.

    Does not raise if thread has no file memory - returns {file_count: 0, file_memory_id: None}.

    Args:
        thread_id: Thread ID to retrieve file metadata for
        user: Required user for permission check

    Returns:
        Dict with file_count and file_memory_id

    Raises:
        ValueError: If thread not found in DB
        PermissionDenied: If user has no VIEW_THREAD permission
    """
    thread = await _get_thread(thread_id)
    if thread is None:
        raise ValueError("Thread not found")
    await _check_object_permission(user, Operation.VIEW_THREAD, thread)

    # FIX: additional query, but we have not yet added it to info
    from django_ai_sdk.conversation.models import Thread
    from django_ai_sdk.memories.models import Entry

    file_memory_id = await (
        Thread.objects.filter(id=thread_id).values_list("file_memory_id", flat=True).afirst()
    )
    file_memory_id_str = str(file_memory_id) if file_memory_id else None
    file_count = (
        await Entry.objects.filter(memory_id=file_memory_id).acount() if file_memory_id_str else 0
    )

    return {
        "file_count": file_count,
        "file_memory_id": file_memory_id_str,
    }


# ============================================================================
# Sync wrappers for use in sync contexts
# ============================================================================

create_thread = async_to_sync(ThreadService.create_thread)
rate_message = async_to_sync(ThreadService.rate_message)
delete_message = async_to_sync(ThreadService.delete_message)
restore_message = async_to_sync(ThreadService.restore_message)
get_thread = async_to_sync(ThreadService.get_thread)
list_threads = async_to_sync(ThreadService.threads)
update_thread = async_to_sync(ThreadService.update_thread)
delete_thread = async_to_sync(ThreadService.delete_thread)
delete_all_threads = async_to_sync(ThreadService.delete_all_threads)
get_thread_file_meta = async_to_sync(aget_thread_file_meta)
