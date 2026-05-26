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
