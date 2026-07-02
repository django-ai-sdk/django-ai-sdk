from __future__ import annotations

from abc import ABC
from enum import StrEnum
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from django.conf import settings
from django.utils.module_loading import import_string

if TYPE_CHECKING:
    from django.db.models import QuerySet

    from django_ai_sdk.assistant import Assistant
    from django_ai_sdk.memories.models import Memory
    from django_ai_sdk.types import UserType


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
    CREATE_ASSISTANT = "create_assistant"
    UPDATE_ASSISTANT = "update_assistant"
    DELETE_ASSISTANT = "delete_assistant"


class BasePermission(ABC):
    async def has_permission(self, user: UserType, operation: Operation, **kwargs: Any) -> bool:
        return True

    async def has_object_permission(
        self,
        user: UserType,
        operation: Operation,
        obj: Any,
        **kwargs: Any,
    ) -> bool:
        return True

    def get_queryset_perms(
        self,
        user: UserType,
        operation: Operation,
        queryset: QuerySet,
    ) -> QuerySet | None:
        """Filter *queryset* to items this class grants *operation* for *user*.

        Implement this to let list endpoints apply your permission rules at the
        database level instead of post-filtering every row.

        Return the filtered ``QuerySet``, ``queryset.none()`` to deny all items,
        or ``None`` to skip this class (no opinion on queryset filtering).
        """
        return None


class AllowAll(BasePermission):
    pass


class DenyAll(BasePermission):
    async def has_permission(self, user: UserType, operation: Operation, **kwargs: Any) -> bool:
        return False

    async def has_object_permission(
        self,
        user: UserType,
        operation: Operation,
        obj: Any,
        **kwargs: Any,
    ) -> bool:
        return False


class IsAuthenticated(BasePermission):
    async def has_permission(self, user: UserType, operation: Operation, **kwargs: Any) -> bool:
        return user is not None and bool(user.is_authenticated)


class IsAdminUser(BasePermission):
    async def has_permission(self, user: UserType, operation: Operation, **kwargs: Any) -> bool:
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

    async def has_permission(self, user: UserType, operation: Operation, **kwargs: Any) -> bool:
        if user is None or not bool(user.is_authenticated):
            return False
        if not hasattr(user, "groups"):
            raise TypeError("IsInAllowedGroups requires a user model with groups support")
        return await user.groups.filter(name__in=self.groups).aexists()  # ty: ignore[unresolved-attribute]


class IsOwner(BasePermission):
    async def has_object_permission(
        self,
        user: UserType,
        operation: Operation,
        obj: Any,
        **kwargs: Any,
    ) -> bool:
        # SECURITY: distinguish "no ownership field" from "anonymous-owned".
        if not hasattr(obj, "user_id"):
            return True  # no opinion on ownership
        owner_id = obj.user_id
        if owner_id is None:
            # Anonymous-owned object: allow only anonymous users.
            return user is None or (hasattr(user, "is_authenticated") and not user.is_authenticated)
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
            Operation.LINK_MEMORY,
            Operation.UNLINK_MEMORY,
        }
    )
    WRITE: frozenset[Operation] = frozenset(
        {
            Operation.UPLOAD_DOCUMENT,
            Operation.DELETE_DOCUMENT,
        }
    )
    MANAGER: frozenset[Operation] = frozenset(
        {
            Operation.UPDATE_MEMORY,
            Operation.DELETE_MEMORY,
        }
    )

    async def has_permission(self, user: UserType, operation: Operation, **kwargs: Any) -> bool:
        return user is not None and bool(user.is_authenticated)

    async def has_object_permission(
        self,
        user: UserType,
        operation: Operation,
        obj: Memory,
        **kwargs: Any,
    ) -> bool:
        if user is None or not bool(user.is_authenticated):
            return False

        ownership = await obj.memory_users.filter(user=user).afirst()
        if ownership is not None:
            if operation in self.MANAGER:
                return ownership.can_manage
            return True

        group_ownership = await obj.memory_groups.filter(group__user=user).afirst()
        if group_ownership is not None:
            if operation in self.MANAGER:
                return group_ownership.can_manage
            return True

        if operation in self.READ and obj.is_public:
            return True
        return False


class AssistantDefaultPermission(BasePermission):
    """Three-tier permission for runtime assistants.

    - Manager (can_manage=True): full access to update/delete.
    - Owner (can_manage=False): can view and use the assistant.
    - Group members: can view and use the assistant.
    - Anonymous: blocked for assistant ops, pass-through for everything else.
    """

    MANAGER: frozenset[Operation] = frozenset(
        {
            Operation.UPDATE_ASSISTANT,
            Operation.DELETE_ASSISTANT,
        }
    )

    ASSISTANT_OPS: frozenset[Operation] = frozenset(
        {
            Operation.VIEW_ASSISTANT,
            Operation.CREATE_ASSISTANT,
            Operation.UPDATE_ASSISTANT,
            Operation.DELETE_ASSISTANT,
        }
    )

    async def has_permission(self, user: UserType, operation: Operation, **kwargs: Any) -> bool:
        if operation not in self.ASSISTANT_OPS:
            return True
        return user is not None and bool(user.is_authenticated)

    async def has_object_permission(
        self,
        user: UserType,
        operation: Operation,
        obj: Any,
        **kwargs: Any,
    ) -> bool:
        # Only handle AssistantSettings objects; pass through for everything else
        from django_ai_sdk.assistants.models import AssistantSettings

        if not isinstance(obj, AssistantSettings):
            return True

        if user is None or not bool(user.is_authenticated):
            return False

        # Check direct user membership
        from django_ai_sdk.assistants.models import AssistantUser

        user_entry = await AssistantUser.objects.filter(assistant=obj, user=user).afirst()
        if user_entry is not None:
            if operation in self.MANAGER:
                return user_entry.can_manage
            return True

        # Check group membership
        from django_ai_sdk.assistants.models import AssistantGroup

        group_entry = await AssistantGroup.objects.filter(assistant=obj, group__user=user).afirst()
        if group_entry is not None:
            if operation in self.MANAGER:
                return group_entry.can_manage
            return True

        return False


@lru_cache(maxsize=1)
def get_default_permissions() -> list[type[BasePermission]]:
    """Resolve default permission classes from settings"""
    paths = getattr(settings, "AI_SDK_DEFAULT_PERMISSIONS", [])
    if not paths:
        return [AssistantDefaultPermission, AllowAll]
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


@lru_cache(maxsize=1)
def get_memory_permissions() -> list[type[BasePermission]]:
    paths = getattr(settings, "AI_SDK_MEMORY_PERMISSIONS", [])
    if not paths:
        return [MemoryDefaultPermission]
    return [import_string(p) for p in paths]


def get_assistant_permissions(assistant: Assistant) -> list[type[BasePermission]]:
    return getattr(assistant, "permissions", get_default_permissions())


async def has_perms(
    user: UserType,
    operation: Operation,
    obj: Any = None,
    *,
    permissions: list[type[BasePermission]] | None = None,
    raise_on_deny: bool = True,
    **kwargs: Any,
) -> bool:
    if permissions is None:
        permissions = get_default_permissions()
    ok = await check_permissions(
        user, operation, permissions, raise_on_deny=raise_on_deny, **kwargs
    )
    if not ok:
        return False
    if obj is not None:
        return await check_object_permissions(
            user, operation, obj, permissions, raise_on_deny=raise_on_deny, **kwargs
        )
    return True


def get_queryset_perms(
    user: UserType,
    operation: Operation,
    queryset: QuerySet,
    *,
    permissions: list[type[BasePermission]] | None = None,
) -> QuerySet:
    if permissions is None:
        permissions = get_default_permissions()
    for cls in permissions:
        perm = ensure_permission_instance(cls)
        result = perm.get_queryset_perms(user, operation, queryset)
        if result is not None:
            queryset = result
    return queryset


async def check_permissions(
    user: UserType,
    operation: Operation,
    permissions: list[type[BasePermission]],
    *,
    raise_on_deny: bool = True,
    **kwargs: Any,
) -> bool:
    for perm_class in permissions:
        perm = ensure_permission_instance(perm_class)
        if not await perm.has_permission(user, operation, **kwargs):
            if raise_on_deny:
                raise PermissionDenied(f"{type(perm).__name__}: {operation.value} not permitted")
            return False
    return True


async def check_object_permissions(
    user: UserType,
    operation: Operation,
    obj: Any,
    permissions: list[type[BasePermission]],
    *,
    raise_on_deny: bool = True,
    **kwargs: Any,
) -> bool:
    for perm_class in permissions:
        perm = ensure_permission_instance(perm_class)
        if not await perm.has_object_permission(user, operation, obj, **kwargs):
            if raise_on_deny:
                raise PermissionDenied(
                    f"{type(perm).__name__}: {operation.value} not permitted for this object"
                )
            return False
    return True
