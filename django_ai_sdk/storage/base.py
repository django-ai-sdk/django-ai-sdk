from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.storage.schemas import ThreadInfo


class StorageType(IntEnum):
    """Lower numbers = faster/cheaper operations"""

    MEMORY = 1
    FILE = 2
    DATABASE = 3
    REST_API = 4


class StorageAdapterRegistry:
    _adapters: ClassVar[dict[type[BaseStorageAdapter], StorageType]] = {}

    @classmethod
    def register(
        cls,
        adapter_class: type[BaseStorageAdapter],
        intensiveness: StorageType,
    ) -> None:
        cls._adapters[adapter_class] = intensiveness

    @classmethod
    def get_all_adapters(cls) -> list[type[BaseStorageAdapter]]:
        """Get all registered adapters sorted by intensiveness (fastest first)."""
        return sorted(cls._adapters.keys(), key=lambda a: cls._adapters[a])

    @classmethod
    def get_intensiveness(cls, adapter_class: type[BaseStorageAdapter]) -> StorageType:
        return cls._adapters.get(adapter_class, StorageType.DATABASE)

    @classmethod
    def clear(cls) -> None:
        cls._adapters.clear()


class BaseStorageAdapter(ABC):
    def __init__(self, thread_id: str) -> None:
        self.thread_id = thread_id

    @classmethod
    @abstractmethod
    async def create_thread(
        cls,
        title: str,
        metadata: dict[str, Any] | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
        thread_id: str | None = None,
    ) -> str:
        pass

    @classmethod
    @abstractmethod
    async def get_thread(cls, thread_id: str) -> ThreadInfo | None:
        pass

    @classmethod
    @abstractmethod
    async def list_threads(
        cls,
        user: AbstractBaseUser | AnonymousUser | None = None,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[ThreadInfo]:
        """limit: Maximum number of threads to return (None = no limit)"""
        pass

    @classmethod
    @abstractmethod
    async def update_thread(
        cls,
        thread_id: str,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """metadata: New metadata dict (optional, merges with existing)
        True if updated successfully, False if thread not found"""
        pass

    @classmethod
    @abstractmethod
    async def delete_thread(cls, thread_id: str) -> bool:
        """True if deleted, False if not found"""
        pass

    @abstractmethod
    async def get_messages(self) -> list[ChatMessage]:
        """Retrieve all ChatMessages for this thread, ordered by creation.
        Should exclude deleted messages."""
        pass

    @abstractmethod
    async def store_chat_message(self, chat_message: ChatMessage) -> str:
        pass

    @abstractmethod
    async def storage_callback(self, chat_message: ChatMessage) -> str | None:
        """Message ID string if stored, None if failed"""
        pass

    @abstractmethod
    async def rate_message(
        self,
        message_id: str,
        rating: int | None,
        feedback: str = "",
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> bool:
        """rating: 1 for good, -1 for bad, or None to unrate
        True if rated successfully, False if message not found"""
        pass

    @abstractmethod
    async def delete_message(self, message_id: str) -> bool:
        """Soft delete a message in this thread.
        True if deleted, False if not found"""
        pass

    @abstractmethod
    async def restore_message(self, message_id: str) -> bool:
        """True if restored, False if not found"""
        pass
