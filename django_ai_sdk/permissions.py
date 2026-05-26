from __future__ import annotations

from abc import ABC
from enum import Enum
from typing import Any


class PermissionDenied(Exception):
    """Raised when a permission check fails."""

    def __init__(self, message: str = "Permission denied") -> None:
        self.message = message
        super().__init__(self.message)


class Operation(str, Enum):
    CHAT = "chat"
    VIEW_HISTORY = "view_history"
    CREATE_THREAD = "create_thread"
    VIEW_THREAD = "view_thread"
    UPDATE_THREAD = "update_thread"
    DELETE_THREAD = "delete_thread"
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


class BasePermission(ABC):
    async def has_permission(self, user: Any, operation: Operation, **kwargs: Any) -> bool:
        return True

    async def has_object_permission(
        self, user: Any, operation: Operation, obj: Any, **kwargs: Any
    ) -> bool:
        return True


class AllowAll(BasePermission):
    pass


class DenyAll(BasePermission):
    async def has_permission(self, user: Any, operation: Operation, **kwargs: Any) -> bool:
        return False

    async def has_object_permission(
        self, user: Any, operation: Operation, obj: Any, **kwargs: Any
    ) -> bool:
        return False


class IsAuthenticated(BasePermission):
    async def has_permission(self, user: Any, operation: Operation, **kwargs: Any) -> bool:
        return user is not None and bool(getattr(user, "is_authenticated", False))


class IsAdminUser(BasePermission):
    async def has_permission(self, user: Any, operation: Operation, **kwargs: Any) -> bool:
        if user is None:
            return False
        return bool(getattr(user, "is_staff", False) or getattr(user, "is_superuser", False))


class IsOwner(BasePermission):
    async def has_object_permission(
        self, user: Any, operation: Operation, obj: Any, **kwargs: Any
    ) -> bool:
        if user is None:
            return False
        user_id = str(getattr(user, "id", user))
        obj_user_id = getattr(obj, "user_id", None)
        if obj_user_id is None:
            return True
        return str(obj_user_id) == user_id


class MemoryDefaultPermission(BasePermission):
    """Three-tier permission for memories.

    - Owner: full access (read, write entries, update/delete memory).
    - Contributor: read + write entries (not update/delete memory).
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
    OWNER: frozenset[Operation] = frozenset(
        {
            Operation.UPDATE_MEMORY,
            Operation.DELETE_MEMORY,
        }
    )

    async def has_permission(self, user: Any, operation: Operation, **kwargs: Any) -> bool:
        return user is not None and bool(user.is_authenticated) | False

    async def has_object_permission(
        self, user: Any, operation: Operation, obj: Any, **kwargs: Any
    ) -> bool:
        if user is None or not bool(user.is_authenticated) | False:
            return False

        owner_id = obj.owner_id
        if owner_id is not None and str(owner_id) == str(user.pk):
            return True

        if operation in self.OWNER:
            return False

        if await obj.contributors.filter(id=user.pk).aexists():
            return True

        if operation in self.READ and obj.is_public:
            return True

        return False


async def check_permissions(
    user: Any, operation: Operation, permissions: list[type[BasePermission]], **kwargs: Any
) -> None:
    for perm_class in permissions:
        perm = perm_class()
        if not await perm.has_permission(user, operation, **kwargs):
            raise PermissionDenied(f"{perm_class.__name__}: {operation.value} not permitted")


async def check_object_permissions(
    user: Any,
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
