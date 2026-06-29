from __future__ import annotations

from abc import ABC
from enum import StrEnum
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from django.conf import settings
from django.utils.module_loading import import_string

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.memories.models import Memory


class PermissionDenied(Exception):
    """Raised when a permission check fails."""

    def __init__(self, message: str = "Permission denied") -> None:
        self.message = message
        super().__init__(self.message)


class Operation(StrEnum):
    CHAT = "chat"
    VIEW_HISTORY = "view_history"
    LIST_THREADS = "list_threads"
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
    async def has_permission(
        self, user: AbstractBaseUser | AnonymousUser | None, operation: Operation, **kwargs: Any
    ) -> bool:
        return True

    async def has_object_permission(
        self,
        user: AbstractBaseUser | AnonymousUser | None,
        operation: Operation,
        obj: Any,
        **kwargs: Any,
    ) -> bool:
        return True


class AllowAll(BasePermission):
    pass


class DenyAll(BasePermission):
    async def has_permission(
        self, user: AbstractBaseUser | AnonymousUser | None, operation: Operation, **kwargs: Any
    ) -> bool:
        return False

    async def has_object_permission(
        self,
        user: AbstractBaseUser | AnonymousUser | None,
        operation: Operation,
        obj: Any,
        **kwargs: Any,
    ) -> bool:
        return False


class IsAuthenticated(BasePermission):
    async def has_permission(
        self, user: AbstractBaseUser | AnonymousUser | None, operation: Operation, **kwargs: Any
    ) -> bool:
        return user is not None and bool(user.is_authenticated)


class IsAdminUser(BasePermission):
    async def has_permission(
        self, user: AbstractBaseUser | AnonymousUser | None, operation: Operation, **kwargs: Any
    ) -> bool:
        return user is not None and bool(
            getattr(user, "is_staff", False) or getattr(user, "is_superuser", False)
        )


class IsInAllowedGroups(BasePermission):
    """Allow only authenticated users belonging to one of the given auth groups.

    Parameterized: instantiate with the allowed group names, e.g.
    ``IsInAllowedGroups(groups=["Sales"])``
    """

    def __init__(self, groups: list[str] | None = None) -> None:
        self.groups = list(groups or [])

    async def has_permission(
        self, user: AbstractBaseUser | AnonymousUser | None, operation: Operation, **kwargs: Any
    ) -> bool:
        if user is None or not bool(user.is_authenticated):
            return False
        return await user.groups.filter(name__in=self.groups).aexists()


class IsOwner(BasePermission):
    async def has_object_permission(
        self,
        user: AbstractBaseUser | AnonymousUser | None,
        operation: Operation,
        obj: Any,
        **kwargs: Any,
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

    async def has_permission(
        self, user: AbstractBaseUser | AnonymousUser | None, operation: Operation, **kwargs: Any
    ) -> bool:
        return user is not None and bool(user.is_authenticated)

    async def has_object_permission(
        self,
        user: AbstractBaseUser | AnonymousUser | None,
        operation: Operation,
        obj: Memory,
        **kwargs: Any,
    ) -> bool:
        if user is None or not bool(user.is_authenticated):
            return False

        ownership = await obj.memory_users.filter(user=user).afirst()
        if ownership is None:
            if operation in self.READ and obj.is_public:
                return True
            return False

        if operation in self.MANAGER:
            return ownership.can_manage
        return True


@lru_cache(maxsize=1)
def get_default_permissions() -> list[type[BasePermission]]:
    paths = getattr(settings, "AI_SDK_DEFAULT_PERMISSIONS", [])
    if not paths:
        return [AllowAll]
    return [import_string(p) for p in paths]


def ensure_permission_instance(
    perm: type[BasePermission] | BasePermission,
) -> BasePermission:
    """Normalise a permission class *or* instance to an instance.

    Callers may store bare classes (e.g. ``[IsAuthenticated]``) or pre-built
    instances (e.g. ``[IsInAllowedGroups(groups=["eng"])]``).  Both forms are
    accepted throughout the permission-checking API; this helper enforces a
    consistent runtime type.
    """
    return perm() if isinstance(perm, type) else perm


async def check_permissions(
    user: AbstractBaseUser | AnonymousUser | None,
    operation: Operation,
    permissions: list[type[BasePermission] | BasePermission],
    **kwargs: Any,
) -> None:
    for perm_class in permissions:
        perm = ensure_permission_instance(perm_class)
        if not await perm.has_permission(user, operation, **kwargs):
            raise PermissionDenied(f"{type(perm).__name__}: {operation.value} not permitted")


async def check_object_permissions(
    user: AbstractBaseUser | AnonymousUser | None,
    operation: Operation,
    obj: Any,
    permissions: list[type[BasePermission] | BasePermission],
    **kwargs: Any,
) -> None:
    for perm_class in permissions:
        perm = ensure_permission_instance(perm_class)
        if not await perm.has_object_permission(user, operation, obj, **kwargs):
            raise PermissionDenied(
                f"{type(perm).__name__}: {operation.value} not permitted for this object"
            )
