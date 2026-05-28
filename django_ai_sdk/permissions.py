from __future__ import annotations

from abc import ABC
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from django.contrib.auth.models import AbstractUser

    from django_ai_sdk.memories.models import Memory


class PermissionDenied(Exception):
    """Raised when a permission check fails."""

    def __init__(self, message: str = "Permission denied") -> None:
        self.message = message
        super().__init__(self.message)


class Operation(StrEnum):
    CHAT = "chat"
    VIEW_HISTORY = "view_history"
    CREATE_THREAD = "create_thread"
    VIEW_THREAD = "view_thread"
    UPDATE_THREAD = "update_thread"
    DELETE_THREAD = "delete_thread"
    DELETE_ALL_THREADS = "delete_all_threads"
    VIEW_MESSAGES = "view_messages"
    SEND_MESSAGE = "send_message"
    RATE_MESSAGE = "rate_message"
    DELETE_MESSAGE = "delete_message"
    RESTORE_MESSAGE = "restore_message"
    UPLOAD_FILE = "upload_file"
    DELETE_FILE = "delete_file"
    VIEW_FILE = "view_file"
    REINDEX = "reindex"
    VIEW_MEMORY = "view_memory"
    CREATE_MEMORY = "create_memory"
    UPDATE_MEMORY = "update_memory"
    DELETE_MEMORY = "delete_memory"
    VIEW_DOCUMENT = "view_document"
    LIST_DOCUMENTS = "list_documents"
    UPLOAD_DOCUMENT = "upload_document"
    DELETE_DOCUMENT = "delete_document"
    LIST_THREAD_MEMORIES = "list_thread_memories"
    LINK_MEMORY = "link_memory"
    UNLINK_MEMORY = "unlink_memory"
    VIEW_ASSISTANT = "view_assistant"


class BasePermission(ABC):
    async def has_permission(self, user: AbstractUser, operation: Operation, **kwargs: Any) -> bool:
        return True

    async def has_object_permission(
        self, user: AbstractUser, operation: Operation, obj: Any, **kwargs: Any
    ) -> bool:
        return True


class AllowAll(BasePermission):
    pass


class DenyAll(BasePermission):
    async def has_permission(self, user: AbstractUser, operation: Operation, **kwargs: Any) -> bool:
        return False

    async def has_object_permission(
        self, user: AbstractUser, operation: Operation, obj: Any, **kwargs: Any
    ) -> bool:
        return False


class IsAuthenticated(BasePermission):
    async def has_permission(self, user: AbstractUser, operation: Operation, **kwargs: Any) -> bool:
        return user is not None and bool(user.is_authenticated)


class IsAdminUser(BasePermission):
    async def has_permission(self, user: AbstractUser, operation: Operation, **kwargs: Any) -> bool:
        return user is not None and bool(user.is_staff or user.is_superuser)


class IsOwner(BasePermission):
    async def has_object_permission(
        self, user: AbstractUser, operation: Operation, obj: Any, **kwargs: Any
    ) -> bool:
        # SECURITY: this seems to deep, we might wanna pass on just the owner.
        owner_id = getattr(obj, "user_id", None)
        if owner_id is None:
            return True
        return user is not None and str(owner_id) == str(user.pk)


class MemoryDefaultPermission(BasePermission):
    """Three-tier permission for memories.

    - Manager (can_manage=True): full access.
    - Owner (can_manage=False): read + write entries (not update/delete memory).
    - Public: read-only (gated by is_public flag on memory).
    - Anonymous: blocked unless public read + global gate allows.
    """

    READ: frozenset[Operation] = frozenset(
        {
            Operation.VIEW_MEMORY,
            Operation.VIEW_DOCUMENT,
            Operation.LIST_DOCUMENTS,
            Operation.LIST_THREAD_MEMORIES,
        }
    )
    WRITE: frozenset[Operation] = frozenset(
        {
            Operation.UPLOAD_DOCUMENT,
            Operation.DELETE_DOCUMENT,
            Operation.LINK_MEMORY,
            Operation.UNLINK_MEMORY,
        }
    )
    MANAGER: frozenset[Operation] = frozenset(
        {
            Operation.UPDATE_MEMORY,
            Operation.DELETE_MEMORY,
        }
    )

    async def has_permission(self, user: AbstractUser, operation: Operation, **kwargs: Any) -> bool:
        return user is not None and bool(user.is_authenticated)

    async def has_object_permission(
        self, user: AbstractUser, operation: Operation, obj: Memory, **kwargs: Any
    ) -> bool:
        if user is None or not bool(user.is_authenticated):
            return False

        ownership = await obj.owners.filter(user=user).afirst()
        if ownership is None:
            if operation in self.READ and obj.is_public:
                return True
            return False

        if operation in self.MANAGER:
            return ownership.can_manage
        return True


async def check_permissions(
    user: AbstractUser,
    operation: Operation,
    permissions: list[type[BasePermission]],
    **kwargs: Any,
) -> None:
    for perm_class in permissions:
        perm = perm_class()
        if not await perm.has_permission(user, operation, **kwargs):
            raise PermissionDenied(f"{perm_class.__name__}: {operation.value} not permitted")


async def check_object_permissions(
    user: AbstractUser,
    operation: Operation,
    obj: Any,
    permissions: list[type[BasePermission]],
    **kwargs: Any,
) -> None:
    for perm_class in permissions:
        perm = perm_class()
        if not await perm.has_object_permission(user, operation, obj, **kwargs):
            raise PermissionDenied(
                f"{perm_class.__name__}: {operation.value} not permitted for this object"
            )
