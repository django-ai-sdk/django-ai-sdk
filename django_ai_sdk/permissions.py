from __future__ import annotations

from abc import ABC
from enum import StrEnum
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

from django.db.models import Q
from django.utils.module_loading import import_string
from pydantic import BaseModel

from django_ai_sdk.utils import resolve_setting

if TYPE_CHECKING:
    from django.db.models import QuerySet

    from django_ai_sdk.agent import Agent
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
    VIEW_AGENT = "view_agent"
    CREATE_AGENT = "create_agent"
    UPDATE_AGENT = "update_agent"
    DELETE_AGENT = "delete_agent"
    USE_INTEGRATION = "use_integration"
    MANAGE_INTEGRATION = "manage_integration"
    VIEW_AUTOMATION = "view_automation"
    RUN_AUTOMATION = "run_automation"
    MANAGE_AUTOMATION = "manage_automation"
    SUBSCRIBE_AUTOMATION = "subscribe_automation"


class PermissionDomain(StrEnum):
    AGENT = "agent"
    THREAD = "thread"
    MEMORY = "memory"
    INTEGRATIONS = "integrations"
    AUTOMATIONS = "automations"


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

        Return the filtered `QuerySet` or `queryset.none()` to deny all items.
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
    `IsInAllowedGroups(groups=["Sales"])`
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

    Parameterized: `IsOwner(field="user_id")` checks `obj.user_id`.
    Use `IsOwner(field="owner_id")` etc. for custom attribute names.
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

    READ: frozenset[Operation] = frozenset(
        {
            Operation.VIEW_THREAD,
            Operation.VIEW_HISTORY,
            Operation.VIEW_MESSAGES,
            Operation.LIST_THREADS,
        }
    )
    WRITE: frozenset[Operation] = frozenset(
        {
            Operation.CHAT,
            Operation.SEND_MESSAGE,
            Operation.UPLOAD_FILE,
            Operation.RATE_MESSAGE,
        }
    )
    MANAGE: frozenset[Operation] = frozenset(
        {
            Operation.DELETE_THREAD,
            Operation.UPDATE_THREAD,
            Operation.DELETE_MESSAGE,
            Operation.RESTORE_MESSAGE,
            Operation.DELETE_FILE,
            Operation.DELETE_ALL_THREADS,
        }
    )

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
    MANAGE: frozenset[Operation] = frozenset(
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

        Applies the same rules as `has_object_permission` but at the
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

        # MANAGE operations — let per-object checks handle the can_manage flag
        return queryset

    async def has_object_permission(
        self,
        user: UserType,
        operation: Operation,
        obj: Memory,
        **kwargs: Any,
    ) -> bool:
        ownership = await obj.memory_users.filter(user=user).afirst()
        if ownership is not None and (operation not in self.MANAGE or ownership.can_manage):
            return True

        group_ownership = await obj.memory_groups.filter(group__user=user).afirst()
        if group_ownership is not None and (
            operation not in self.MANAGE or group_ownership.can_manage
        ):
            return True

        if operation in self.READ and obj.is_public:
            return True
        return False


# Not a class attribute: get_object_permissions_map() auto-discovers upper-case
# frozensets as READ/WRITE/MANAGE-style tiers, and this is dispatch, not a tier.
_USE_OPERATIONS: frozenset[Operation] = frozenset(
    {
        Operation.CHAT,
        Operation.VIEW_THREAD,
        Operation.UPLOAD_FILE,
        Operation.VIEW_FILE,
        Operation.DELETE_FILE,
    }
)


class AgentDefaultPermission(BasePermission):
    """Three-tier permission for agents.

    - Manager (can_manage=True): full access to update/delete.
    - Owner (can_manage=False): can view and use the agent.
    - Group members: can view and use the agent.
    - Public (is_public on the row): read and use, never manage.
    - Anonymous: blocked for agent ops, pass-through for everything else.

    Administering an agent carries the row as `obj`; using one has no such object,
    so there the row arrives as the `agent` keyword the agent services pass.
    """

    READ: frozenset[Operation] = frozenset(
        {
            Operation.VIEW_AGENT,
        }
    )
    WRITE: frozenset[Operation] = frozenset(
        {
            Operation.CREATE_AGENT,
        }
    )
    MANAGE: frozenset[Operation] = frozenset(
        {
            Operation.UPDATE_AGENT,
            Operation.DELETE_AGENT,
        }
    )

    AGENT_OPS: frozenset[Operation] = frozenset(READ | WRITE | MANAGE)

    async def has_permission(self, user: UserType, operation: Operation, **kwargs: Any) -> bool:
        if operation in _USE_OPERATIONS:
            if "agent" not in kwargs:
                return False

            agent = kwargs["agent"]
            config = agent.config if agent.is_runtime else None

            if config is None:
                return True  # code-related, so no gate
            if config.is_public:
                return True
            if user is None or not bool(user.is_authenticated):
                return False

            return await self._membership_allows(user, operation, config)
        if operation not in self.AGENT_OPS:
            return True
        return user is not None and bool(user.is_authenticated)

    async def _membership_allows(self, user: UserType, operation: Operation, config: Any) -> bool:
        from django_ai_sdk.agents.models import AgentGroup, AgentUser

        user_entry = await AgentUser.objects.filter(agent=config, user=user).afirst()
        if user_entry is not None and (operation not in self.MANAGE or user_entry.can_manage):
            return True
        group_entry = await AgentGroup.objects.filter(agent=config, group__user=user).afirst()
        return group_entry is not None and (operation not in self.MANAGE or group_entry.can_manage)

    async def has_object_permission(
        self,
        user: UserType,
        operation: Operation,
        obj: Any,
        **kwargs: Any,
    ) -> bool:
        # Only handle AgentSettings objects; pass through for everything else
        from django_ai_sdk.agents.models import AgentSettings

        if not isinstance(obj, AgentSettings):
            return True

        if operation in self.READ and obj.is_public:
            return True
        if user is None or not bool(user.is_authenticated):
            return False
        return await self._membership_allows(user, operation, obj)


class IntegrationDefaultPermission(BasePermission):
    """Default permission for integrations: any authenticated user may use them.

    Override per-integration via `Integration.permissions` or globally via
    `AI_SDK_PERMISSIONS["integrations"]`.
    """

    async def has_permission(self, user: UserType, operation: Operation, **kwargs: Any) -> bool:
        return user is not None and bool(user.is_authenticated)


class AutomationDefaultPermission(BasePermission):
    """Read and self-subscribe for any authenticated user; run and manage for staff.

    Enabling or firing an automation acts on every other user; subscribing does not.
    """

    READ = frozenset({Operation.VIEW_AUTOMATION, Operation.SUBSCRIBE_AUTOMATION})

    async def has_permission(self, user: UserType, operation: Operation, **kwargs: Any) -> bool:
        if user is None or not user.is_authenticated:
            return False
        if operation in self.READ:
            return True
        return bool(getattr(user, "is_staff", False) or getattr(user, "is_superuser", False))


DOMAIN_PERMISSION_DEFAULTS: dict[PermissionDomain, list[str]] = {
    PermissionDomain.AGENT: ["django_ai_sdk.permissions.AgentDefaultPermission"],
    PermissionDomain.THREAD: ["django_ai_sdk.permissions.ThreadDefaultPermission"],
    PermissionDomain.MEMORY: ["django_ai_sdk.permissions.MemoryDefaultPermission"],
    PermissionDomain.INTEGRATIONS: ["django_ai_sdk.permissions.IntegrationDefaultPermission"],
    PermissionDomain.AUTOMATIONS: ["django_ai_sdk.permissions.AutomationDefaultPermission"],
}


@lru_cache(maxsize=10)
def get_domain_permissions(domain: PermissionDomain) -> list[type[BasePermission]]:
    """Resolve default permission classes for a permission domain.

    Resolution:
    1. AI_SDK_PERMISSIONS setting (per-domain override), or
    2. Built-in DOMAIN_PERMISSION_DEFAULTS (defaults)
    """
    overrides = resolve_setting("AI_SDK_PERMISSIONS", {}).get(domain.value)
    if overrides is not None:
        return [import_string(p) for p in overrides] if overrides else []
    paths = DOMAIN_PERMISSION_DEFAULTS.get(domain, [])
    return [import_string(p) for p in paths]


class PermissionsMixin:
    """Mixin providing permission helpers for stateless service classes.
    Subclasses must set `domain` to a `PermissionDomain` value.
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
    async def get_object_permissions_map(
        cls,
        user: UserType,
        obj: Any,
    ) -> dict[str, bool]:
        """Auto-discover frozenset permission groups and check user against them.

        Iterates the domain's permission classes, finds `READ` / `WRITE` /
        `MANAGE` (and any other upper-case frozensets), and checks one
        representative operation per group against the full permission chain.

        Returns a dict like `{"read": True, "write": False, "manage": False}`.
        """
        groups: dict[str, Operation] = {}
        for perm_cls in cls.get_default_permissions():
            perm = ensure_permission_instance(perm_cls)
            for attr in dir(perm):
                val = getattr(perm, attr, None)
                if attr.isupper() and isinstance(val, frozenset) and val:
                    key = attr.lower()
                    if key not in groups:
                        groups[key] = cast("Operation", next(iter(val)))
        result: dict[str, bool] = {}
        for name, op in groups.items():
            try:
                await cls.has_perms(user, op, obj, raise_on_deny=True)
                result[name] = True
            except PermissionDenied:
                result[name] = False
        return result

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

    Callers may store bare classes (e.g. `[IsAuthenticated]`) or pre-built
    instances (e.g. `[IsInAllowedGroups(groups=["eng"])]`).  Both forms are
    accepted throughout the permission-checking API; this helper enforces a
    consistent runtime type.
    """
    return perm() if isinstance(perm, type) else perm


def get_agent_permissions(agent: Agent | None) -> list[type[BasePermission]]:
    """`agent.permissions` if non-empty, else the AGENT domain default.

    An empty list falls back rather than disabling every check, matching
    get_integration_permissions and get_automation_permissions. An agent that really
    wants no gating says so with [AllowAll].
    """
    perms = getattr(agent, "permissions", None) if agent is not None else None
    if perms:
        return perms
    return get_domain_permissions(PermissionDomain.AGENT)


def get_integration_permissions(service: Any) -> list[type[BasePermission]]:
    """Resolve perms for an integration service.

    Resolution:
    1. `service.permissions` if non-empty (per-integration override)
    2. Domain default for INTEGRATIONS (fallback)
    """
    perms = getattr(service, "permissions", None)
    if perms:
        return perms
    return get_domain_permissions(PermissionDomain.INTEGRATIONS)


def get_automation_permissions(automation: Any) -> list[type[BasePermission]]:
    """`automation.permissions` if set, else the AUTOMATIONS domain default."""
    perms = getattr(automation, "permissions", None)
    if perms:
        return perms
    return get_domain_permissions(PermissionDomain.AUTOMATIONS)


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
        permissions = get_domain_permissions(PermissionDomain.AGENT)
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


class ObjectPermissions(BaseModel):
    can_read: bool = False
    can_write: bool = False
    can_manage: bool = False
