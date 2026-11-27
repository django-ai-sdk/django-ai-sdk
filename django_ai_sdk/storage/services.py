from __future__ import annotations

from typing import TYPE_CHECKING, Any

from asgiref.sync import async_to_sync

from django_ai_sdk.agents.services import AgentService
from django_ai_sdk.errors import AiSdkError, ErrorCode, NotFound
from django_ai_sdk.logger import get_logger
from django_ai_sdk.permissions import (
    Operation,
    PermissionDenied,
    PermissionDomain,
    PermissionsMixin,
)
from django_ai_sdk.storage.base import StorageAdapterRegistry

if TYPE_CHECKING:
    from django_ai_sdk.agents.base import Agent
    from django_ai_sdk.storage.schemas import ThreadDetail, ThreadInfo
    from django_ai_sdk.types import UserType

logger = get_logger(__name__)


def thread_metadata(agent_id: str, agent: Agent, created_via: str) -> dict[str, Any]:
    return {
        "agent_id": agent_id,
        "model": agent.model,
        "agent_name": agent.name or agent.__class__.__name__,
        "agent_class": agent.__class__.__name__,
        "created_via": created_via,
    }


async def _get_thread(thread_id: str) -> ThreadInfo | None:
    for adapter_class in StorageAdapterRegistry.get_all_adapters():
        thread = await adapter_class.get_thread(thread_id)
        if thread:
            return thread
    return None


async def _get_storage(thread_info: ThreadInfo) -> Any:
    agent = await AgentService.get(thread_info.agent_id)
    storage_class = ThreadService.get_storage(agent)
    return storage_class(thread_info.id)


class ThreadService(PermissionsMixin):
    """All async; use the sync-prefixed aliases in sync contexts."""

    domain = PermissionDomain.THREAD

    @classmethod
    async def create_thread(
        cls,
        agent_id: str,
        title: str = "",
        metadata: dict | None = None,
        *,
        user: UserType,
        thread_id: str | None = None,
    ) -> str:
        """`metadata` overrides the auto-filled model/agent fields. Needs CREATE_THREAD."""
        from django_ai_sdk.agents.services import AgentService

        agent = await AgentService.get(agent_id)

        await cls.has_perms(user, Operation.CREATE_THREAD)

        storage_class = agent.storage_adapter
        title = title or ""
        default_metadata = thread_metadata(agent_id, agent, "create_thread")
        default_metadata.update(metadata or {})

        thread_id = await storage_class.create_thread(
            title=title,
            metadata=default_metadata,
            user=user,
            thread_id=thread_id,
        )

        logger.debug(f"Created thread {thread_id} for agent {agent_id}")
        return thread_id

    @classmethod
    async def get_thread(cls, thread_id: str, *, user: UserType) -> ThreadInfo | None:
        """Find a thread across storage adapters; None if not found. Needs VIEW_THREAD."""
        thread = await _get_thread(thread_id)
        if thread:
            await cls.has_perms(user, Operation.VIEW_THREAD, thread)
            logger.debug(f"Found thread {thread_id} in adapter, agent: {thread.agent_id}")

        if not thread:
            logger.debug(f"Thread not found: {thread_id}")

        return thread

    @classmethod
    async def threads(
        cls,
        user: UserType,
        *,
        limit: int | None = 100,
        offset: int = 0,
    ) -> list[ThreadInfo]:
        """List threads from all storage adapters; `limit=None` returns all."""
        all_threads: list[ThreadInfo] = []

        for adapter_class in StorageAdapterRegistry.get_all_adapters():
            threads = await adapter_class.list_threads(user)
            for thread in threads:
                try:
                    await cls.has_perms(user, Operation.LIST_THREADS, thread)
                    all_threads.append(thread)
                except PermissionDenied:
                    continue
            logger.debug(f"Found {len(threads)} threads in {adapter_class.__name__}")

        all_threads.sort(key=lambda t: t.updated_at, reverse=True)

        logger.debug(f"Total threads: {len(all_threads)}")
        return all_threads[offset : offset + limit if limit is not None else None]

    @classmethod
    async def update_thread(
        cls,
        thread_id: str,
        *,
        user: UserType,
        title: str | None = None,
        metadata: dict | None = None,
    ) -> bool:
        """Update a thread's title or merge its metadata; False if not found."""
        thread = await _get_thread(thread_id)
        if thread:
            await cls.has_perms(user, Operation.UPDATE_THREAD, thread)

        for adapter_class in StorageAdapterRegistry.get_all_adapters():
            success = await adapter_class.update_thread(thread_id, title, metadata)
            if success:
                logger.debug(f"Updated thread {thread_id} in {adapter_class.__name__}")
                return True

        return False

    @classmethod
    async def delete_thread(cls, thread_id: str, *, user: UserType) -> bool:
        """Delete a thread and its messages; False if not found."""
        thread = await _get_thread(thread_id)
        if thread:
            await cls.has_perms(user, Operation.DELETE_THREAD, thread)

        for adapter_class in StorageAdapterRegistry.get_all_adapters():
            success = await adapter_class.delete_thread(thread_id)
            if success:
                logger.debug(f"Deleted thread {thread_id} from {adapter_class.__name__}")
                return True

        return False

    @classmethod
    async def delete_all_threads(cls, *, user: UserType) -> int:
        """Delete the user's threads and their messages; returns how many."""
        total_deleted = 0
        for adapter_class in StorageAdapterRegistry.get_all_adapters():
            threads = await adapter_class.list_threads(user)
            for thread in threads:
                try:
                    await cls.has_perms(user, Operation.DELETE_ALL_THREADS, thread)
                    if await adapter_class.delete_thread(thread.id):
                        total_deleted += 1
                except PermissionDenied:
                    continue
            logger.debug(f"Deleted threads from {adapter_class.__name__}")

        return total_deleted

    @classmethod
    async def rate_message(
        cls,
        thread_id: str,
        message_id: str,
        rating: int | None,
        feedback: str = "",
        *,
        user: UserType,
    ) -> bool:
        """Rate a message: 1 good, -1 bad. Raises ValueError if not found."""
        thread = await _get_thread(thread_id)
        if not thread:
            raise NotFound("Thread not found")
        await cls.has_perms(user, Operation.RATE_MESSAGE, thread)

        storage = await _get_storage(thread)
        success = await storage.rate_message(message_id, rating, feedback, user=user)
        if not success:
            raise NotFound("Message not found")
        return True

    @classmethod
    async def delete_message(cls, thread_id: str, message_id: str, *, user: UserType) -> bool:
        """Soft delete a message. Raises ValueError if not found."""
        thread = await _get_thread(thread_id)
        if not thread:
            raise NotFound("Thread not found")
        await cls.has_perms(user, Operation.DELETE_MESSAGE, thread)

        storage = await _get_storage(thread)
        success = await storage.delete_message(message_id)
        if not success:
            raise NotFound("Message not found")
        return True

    @classmethod
    async def restore_message(cls, thread_id: str, message_id: str, *, user: UserType) -> bool:
        """Restore a soft-deleted message. Raises ValueError if not found."""
        thread = await _get_thread(thread_id)
        if not thread:
            raise NotFound("Thread not found")
        await cls.has_perms(user, Operation.RESTORE_MESSAGE, thread)

        storage = await _get_storage(thread)
        success = await storage.restore_message(message_id)
        if not success:
            raise NotFound("Message not found")
        return True

    @classmethod
    async def storage_for_thread(cls, thread_id: str, *, user: UserType) -> Any:
        """Raises ValueError if the thread does not exist or has no storage."""
        thread = await _get_thread(thread_id)
        if thread is None:
            raise NotFound(f"Thread not found: {thread_id}")
        await cls.has_perms(user, Operation.VIEW_THREAD, thread)
        return await _get_storage(thread)

    @classmethod
    def get_storage(cls, agent: Any) -> type:
        storage_class = agent.storage_adapter
        if storage_class is None:
            raise AiSdkError(
                f"Agent '{agent.agent_id}' has no storage_adapter configured",
                ErrorCode.CONFIGURATION_ERROR,
            )
        return storage_class


async def aget_thread_history(thread_id: str, user: UserType) -> dict[str, Any]:
    from django_ai_sdk.agents.services import AgentService

    agent = await AgentService.get_agent(thread_id, user=user)
    thread_detail: ThreadDetail = await agent.history(thread_id, user=user)

    return {
        "thread": thread_detail.thread,
        "messages": thread_detail.messages,
    }


get_thread_history = async_to_sync(aget_thread_history)


def keep_own_feedback(history: dict[str, Any], user: UserType) -> None:
    user_pk = str(user.pk) if user and user.is_authenticated else None
    for message in history.get("messages", []):
        feedbacks = message.pop("feedbacks", [])
        message["feedback"] = next((fb for fb in feedbacks if fb.get("user_id") == user_pk), None)


async def aget_thread_file_meta(thread_id: str, *, user: UserType) -> dict[str, Any]:
    """Return a thread's file_count and file_memory_id (0 and None without a file memory)."""
    thread = await _get_thread(thread_id)
    if thread is None:
        raise NotFound("Thread not found")
    await ThreadService.has_perms(user, Operation.VIEW_THREAD, thread)

    from django_ai_sdk.memories.models import Entry  # noqa: PLC0415

    file_memory_id = thread.file_memory_id
    file_count = (
        await Entry.objects.filter(memory_id=file_memory_id).acount() if file_memory_id else 0
    )

    return {
        "file_count": file_count,
        "file_memory_id": file_memory_id,
    }


async def aget_thread_sources(thread_id: str, *, user: UserType) -> list[dict[str, Any]]:
    thread = await _get_thread(thread_id)
    if thread is None:
        raise NotFound("Thread not found")
    await ThreadService.has_perms(user, Operation.VIEW_THREAD, thread)

    from django_ai_sdk.artifacts.models import Artifact  # noqa: PLC0415
    from django_ai_sdk.artifacts.schemas import (  # noqa: PLC0415
        CitationsArtifact,
        CitedSource,
    )

    sources: dict[str, dict[str, Any]] = {}
    rows = Artifact.objects.filter(
        thread_id=thread_id, schema_name=CitationsArtifact.__name__
    ).order_by("-created_at")
    async for row in rows:
        for source in map(CitedSource.model_validate, row.data.get("sources", [])):
            known = sources.setdefault(
                source.key,
                source.model_dump(mode="json", by_alias=True, exclude={"number"}),
            )
            known["cited"] = known["cited"] or source.cited
            known["content"] = known["content"] or source.content
    return list(sources.values())


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
get_thread_sources = async_to_sync(aget_thread_sources)
