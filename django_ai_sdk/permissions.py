from __future__ import annotations

from abc import ABC
from enum import StrEnum
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from django.conf import settings
from django.db.models import Q
from django.utils.module_loading import import_string

if TYPE_CHECKING:
    from django.db.models import QuerySet

    from django_ai_sdk.assistant import Assistant
    from django_ai_sdk.memories.models import Memory
    from django_ai_sdk.storage.schemas import ThreadInfo
    from django_ai_sdk.types import UserType


class PermissionDenied(Exception):
    """Raised when a permission check fails."""

    def __init__(self, message: str = "Permission denied") -> None:
        self.message = message
        super().__init__(self.message)


class ConflictError(Exception):
    """Raised when an operation conflicts with existing state."""

    def __init__(self, message: str = "Conflict") -> None:
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
    USE_INTEGRATION = "use_integration"
    MANAGE_INTEGRATION = "manage_integration"


class PermissionDomain(StrEnum):
    ASSISTANT = "assistant"
    THREAD = "thread"
    MEMORY = "memory"
    INTEGRATIONS = "integrations"


class BasePermission(ABC):
    async def has_permission(self, user: UserType, operation: Operation, **kwargs: Any) -> bool:
        """Check if the user has permission for the given operation."""
        return True

    async def has_object_permission(
        self,
        user: UserType,
        operation: Operation,
        obj: Any,
        **kwargs: Any,
    ) -> bool:
        """Check if the user has permission for the given operation on the object."""
        return True

    def get_queryset_perms(
        self,
        user: UserType,
        operation: Operation,
        queryset: QuerySet,
    ) -> QuerySet:
        """Filter *queryset* to items this class grants *operation* for *user*.

        Implement this to let list endpoints apply your permission rules at the
        database level instead of post-filtering every row.

        Return the filtered ``QuerySet`` or ``queryset.none()`` to deny all items.
        The default implementation returns the queryset unchanged.
        """
        return queryset


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
    """Allow only the authenticated owner of an object.

    Parameterized: ``IsOwner(field="user_id")`` checks ``obj.user_id``.
    Use ``IsOwner(field="owner_id")`` etc. for custom attribute names.
    """

    def __init__(self, field: str = "user_id") -> None:
        self.field = field

    async def has_object_permission(
        self,
        user: UserType,
        operation: Operation,
        obj: Any,
        **kwargs: Any,
    ) -> bool:
        # SECURITY: distinguish "no ownership field" from "anonymous-owned".
        owner_id = getattr(obj, self.field, None)
        if owner_id is None:
            return False
        return user is not None and bool(user.is_authenticated) and str(owner_id) == str(user.pk)


class ThreadDefaultPermission(BasePermission):
    """Default permission class for threads.

    - Authenticated users: full access to their own threads.
    - Anonymous users: no access to threads.
    """

    async def has_permission(self, user: UserType, operation: Operation, **kwargs: Any) -> bool:
        # Only authenticated users can access threads
        if user is None or not user.is_authenticated:
            return False
        return True

    async def has_object_permission(
        self,
        user: UserType,
        operation: Operation,
        obj: ThreadInfo,
        **kwargs: Any,
    ) -> bool:
        if user is None or not user.is_authenticated:
            return False
        return obj.user_id is not None and str(obj.user_id) == str(getattr(user, "pk", ""))


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

    def get_queryset_perms(
        self,
        user: UserType,
        operation: Operation,
        queryset: QuerySet,
    ) -> QuerySet:
        """Filter *queryset* to memories this user may access for *operation*.

        Applies the same rules as ``has_object_permission`` but at the
        database level so list views avoid iterating every row.
        """
        if user is None or not bool(user.is_authenticated):
            return queryset.none()

        if operation in self.READ:
            return queryset.filter(
                Q(is_public=True) | Q(memory_users__user=user) | Q(memory_groups__group__user=user)
            ).distinct()

        if operation in self.WRITE:
            return queryset.filter(
                Q(memory_users__user=user) | Q(memory_groups__group__user=user)
            ).distinct()

        # MANAGER operations — let per-object checks handle the can_manage flag
        return queryset

    async def has_object_permission(
        self,
        user: UserType,
        operation: Operation,
        obj: Memory,
        **kwargs: Any,
    ) -> bool:
        ownership = await obj.memory_users.filter(user=user).afirst()
        if ownership is not None and (operation not in self.MANAGER or ownership.can_manage):
            return True

        group_ownership = await obj.memory_groups.filter(group__user=user).afirst()
        if group_ownership is not None and (
            operation not in self.MANAGER or group_ownership.can_manage
        ):
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
        if user_entry is not None and (operation not in self.MANAGER or user_entry.can_manage):
            return True

        # Check group membership
        from django_ai_sdk.assistants.models import AssistantGroup

        group_entry = await AssistantGroup.objects.filter(assistant=obj, group__user=user).afirst()
        if group_entry is not None and (operation not in self.MANAGER or group_entry.can_manage):
            return True

        return False


class IntegrationDefaultPermission(BasePermission):
    """Default permission for integrations: any authenticated user may use them.

    Override per-integration via ``Integration.permissions`` or globally via
    ``AI_SDK_PERMISSIONS["integrations"]``.
    """

    async def has_permission(self, user: UserType, operation: Operation, **kwargs: Any) -> bool:
        return user is not None and bool(user.is_authenticated)


DOMAIN_PERMISSION_DEFAULTS: dict[PermissionDomain, list[str]] = {
    PermissionDomain.ASSISTANT: ["django_ai_sdk.permissions.AssistantDefaultPermission"],
    PermissionDomain.THREAD: ["django_ai_sdk.permissions.ThreadDefaultPermission"],
    PermissionDomain.MEMORY: ["django_ai_sdk.permissions.MemoryDefaultPermission"],
    PermissionDomain.INTEGRATIONS: ["django_ai_sdk.permissions.IntegrationDefaultPermission"],
}


@lru_cache(maxsize=10)
def get_domain_permissions(domain: PermissionDomain) -> list[type[BasePermission]]:
    """Resolve default permission classes for a permission domain.

    Resolution:
    1. AI_SDK_PERMISSIONS setting (per-domain override), or
    2. Built-in DOMAIN_PERMISSION_DEFAULTS (defaults)
    """
    overrides = getattr(settings, "AI_SDK_PERMISSIONS", {}).get(domain.value)
    if overrides is not None:
        return [import_string(p) for p in overrides] if overrides else []
    paths = DOMAIN_PERMISSION_DEFAULTS.get(domain, [])
    return [import_string(p) for p in paths]


class PermissionsMixin:
    """Mixin providing permission helpers for stateless service classes.
    Subclasses must set ``domain`` to a :class:`PermissionDomain` value.
    """

    domain: PermissionDomain

    @classmethod
    def get_default_permissions(cls) -> list[type[BasePermission]]:
        """Return the default permission classes for this service's domain."""
        if not hasattr(cls, "domain"):
            raise AttributeError(f"{cls.__name__} must define a 'domain' attribute")
        return get_domain_permissions(cls.domain)

    @classmethod
    async def has_perms(
        cls,
        user: UserType,
        operation: Operation,
        obj: Any = None,
        *,
        raise_on_deny: bool = True,
        **kwargs: Any,
    ) -> bool:
        perms = cls.get_default_permissions()
        return await has_perms(
            user, operation, obj, permissions=perms, raise_on_deny=raise_on_deny, **kwargs
        )

    @classmethod
    def has_queryset_perms(
        cls,
        user: UserType,
        operation: Operation,
        *,
        queryset: QuerySet,
        permissions: list[type[BasePermission]] | None = None,
    ) -> QuerySet:
        """Filter a queryset through the domain's permission classes.

        Each permission class in the domain chain receives the queryset and
        can return a filtered version.  Classes that don't know how to filter
        this model pass it through unchanged.
        """
        perms = permissions if permissions is not None else cls.get_default_permissions()
        for perm_cls in perms:
            perm = ensure_permission_instance(perm_cls)
            queryset = perm.get_queryset_perms(user, operation, queryset)
        return queryset


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


def get_assistant_permissions(assistant: Assistant | None) -> list[type[BasePermission]]:
    """Resolve perms for an assistant.

    Resolution:
    1. assistant.permissions if set (per-assistant override)
    2. Domain default for ASSISTANT (fallback)
    """
    perms = getattr(assistant, "permissions", None) if assistant is not None else None
    if perms is not None:
        return perms
    return get_domain_permissions(PermissionDomain.ASSISTANT)


def get_integration_permissions(service: Any) -> list[type[BasePermission]]:
    """Resolve perms for an integration service.

    Resolution:
    1. ``service.permissions`` if non-empty (per-integration override)
    2. Domain default for INTEGRATIONS (fallback)
    """
    perms = getattr(service, "permissions", None)
    if perms:
        return perms
    return get_domain_permissions(PermissionDomain.INTEGRATIONS)


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
        permissions = get_domain_permissions(PermissionDomain.ASSISTANT)
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
