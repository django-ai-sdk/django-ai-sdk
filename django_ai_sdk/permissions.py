from __future__ import annotations

from abc import ABC
from enum import StrEnum
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from django.db.models import Q
from django.utils.module_loading import import_string
from pydantic import BaseModel

from django_ai_sdk.utils import resolve_setting

if TYPE_CHECKING:
    from django.db.models import QuerySet

    from django_ai_sdk.agents.base import Agent
    from django_ai_sdk.memories.models import Memory
    from django_ai_sdk.storage.schemas import ThreadInfo
    from django_ai_sdk.types import UserType


class PermissionDenied(Exception):
    def __init__(self, message: str = "Permission denied") -> None:
        self.message = message
        super().__init__(self.message)


class ConflictError(Exception):
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
    VIEW_WORKFLOW = "view_workflow"
    RUN_WORKFLOW = "run_workflow"
    MANAGE_WORKFLOW = "manage_workflow"
    APPROVE_WORKFLOW = "approve_workflow"
    VIEW_OVERSIGHT_METRICS = "view_oversight_metrics"
    VIEW_OVERSIGHT_CONTENT = "view_oversight_content"
    REVIEW_OVERSIGHT = "review_oversight"
    VIEW_OVERSIGHT_AUTHORING = "view_oversight_authoring"
    AUTHOR_OVERSIGHT = "author_oversight"


class PermissionDomain(StrEnum):
    AGENT = "agent"
    THREAD = "thread"
    MEMORY = "memory"
    INTEGRATIONS = "integrations"
    WORKFLOW = "workflow"
    OVERSIGHT = "oversight"


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
    ) -> QuerySet:
        """Default: the queryset unchanged."""
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
    """Allow only authenticated users belonging to one of the given auth groups."""

    def __init__(self, groups: list[str] | None = None) -> None:
        self.groups = list(groups or [])

    async def has_permission(self, user: UserType, operation: Operation, **kwargs: Any) -> bool:
        if user is None or not bool(user.is_authenticated):
            return False
        if not hasattr(user, "groups"):
            raise TypeError("IsInAllowedGroups requires a user model with groups support")
        return await user.groups.filter(name__in=self.groups).aexists()  # ty: ignore[missing-attribute]


class IsOwner(BasePermission):
    """Allow only the authenticated owner: ``IsOwner(field="user_id")`` checks ``obj.user_id``."""

    def __init__(self, field: str = "user_id") -> None:
        self.field = field

    async def has_object_permission(
        self,
        user: UserType,
        operation: Operation,
        obj: Any,
        **kwargs: Any,
    ) -> bool:
        owner_id = getattr(obj, self.field, None)
        if owner_id is None:
            return False
        return user is not None and bool(user.is_authenticated) and str(owner_id) == str(user.pk)


class ThreadDefaultPermission(BasePermission):
    """- Authenticated users: full access to their own threads.
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
        }
    )
    DELETE: frozenset[Operation] = frozenset(
        {
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
        """Filter *queryset* by the ``has_object_permission`` rules, in the database."""
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

        return queryset

    async def has_object_permission(
        self,
        user: UserType,
        operation: Operation,
        obj: Memory,
        **kwargs: Any,
    ) -> bool:

        restricted = self.MANAGE | self.DELETE

        ownership = await obj.memory_users.filter(user=user).afirst()  # ty: ignore[invalid-attribute-access]
        if ownership is not None and (operation not in restricted or ownership.can_manage):
            return True

        group_ownership = await obj.memory_groups.filter(group__user=user).afirst()  # ty: ignore[invalid-attribute-access]
        if group_ownership is not None and (
            operation not in restricted or group_ownership.can_manage
        ):
            return True

        if operation in self.READ and obj.is_public:
            return True
        return False


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
    """Administering an agent carries the row as `obj`; using one has no such object,
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

    async def has_permission(self, user: UserType, operation: Operation, **kwargs: Any) -> bool:
        if operation in _USE_OPERATIONS:
            if "agent" not in kwargs:
                return False

            agent = kwargs["agent"]
            config = agent.config if agent.is_runtime else None

            if config is None:
                return True
            if config.is_public:
                return True
            if user is None or not bool(user.is_authenticated):
                return False

            return await self._membership_allows(user, operation, config)
        if operation not in self.READ | self.WRITE | self.MANAGE:
            return True
        return user is not None and bool(user.is_authenticated)

    async def _membership_allows(self, user: UserType, operation: Operation, config: Any) -> bool:
        from django_ai_sdk.agents.models import AgentGroup, AgentUser

        user_entry = await AgentUser.objects.filter(agent=config, user=user).afirst()
        if user_entry is not None and (operation not in self.MANAGE or user_entry.can_manage):
            return True
        agent_groups = AgentGroup.objects.filter(agent=config, group__user=user)
        if operation not in self.MANAGE:
            return await agent_groups.aexists()
        return await agent_groups.filter(can_manage=True).aexists()

    async def has_object_permission(
        self,
        user: UserType,
        operation: Operation,
        obj: Any,
        **kwargs: Any,
    ) -> bool:
        from django_ai_sdk.agents.models import AgentSettings

        if not isinstance(obj, AgentSettings):
            return True

        if operation in self.READ and obj.is_public:
            return True
        if user is None or not bool(user.is_authenticated):
            return False
        return await self._membership_allows(user, operation, obj)


class IntegrationDefaultPermission(BasePermission):
    """Default permission for integrations: any authenticated user may use them."""

    async def has_permission(self, user: UserType, operation: Operation, **kwargs: Any) -> bool:
        return user is not None and bool(user.is_authenticated)


class WorkflowDefaultPermission(BasePermission):
    """Any authenticated user may run a workflow. Authoring needs a model permission."""

    READ: frozenset[Operation] = frozenset({Operation.VIEW_WORKFLOW})
    WRITE: frozenset[Operation] = frozenset({Operation.RUN_WORKFLOW})
    MANAGE: frozenset[Operation] = frozenset({Operation.MANAGE_WORKFLOW})

    MANAGE_PERMISSION = "django_ai_sdk_workflows.change_workflow"
    VIEW_RUNS_PERMISSION = "django_ai_sdk_workflows.view_workflowrun"
    APPROVE_PERMISSION = "django_ai_sdk_workflows.approve_steprun"

    async def has_permission(self, user: UserType, operation: Operation, **kwargs: Any) -> bool:
        if user is None or not bool(user.is_authenticated):
            return False
        if operation in self.MANAGE:
            return await _ahas_perm(user, self.MANAGE_PERMISSION)
        if operation is Operation.APPROVE_WORKFLOW:
            return await _ahas_perm(user, self.APPROVE_PERMISSION)
        return operation in self.READ | self.WRITE

    async def has_object_permission(
        self,
        user: UserType,
        operation: Operation,
        obj: Any,
        **kwargs: Any,
    ) -> bool:
        from django_ai_sdk.workflows.models import Workflow, WorkflowRun

        if user is None or not bool(user.is_authenticated):
            return False
        if isinstance(obj, Workflow):
            if operation in self.MANAGE:
                return await _ahas_perm(user, self.MANAGE_PERMISSION)
            return operation in self.READ | self.WRITE
        if isinstance(obj, WorkflowRun):
            if operation is Operation.APPROVE_WORKFLOW:
                own = _owned_by(obj.user_id, user) or _owned_by(obj.started_by_id, user)
                return not own and await _ahas_perm(user, self.APPROVE_PERMISSION)
            if _owned_by(obj.user_id, user):
                return True
            return operation in self.READ and await _ahas_perm(user, self.VIEW_RUNS_PERMISSION)
        return False

    def get_queryset_perms(
        self,
        user: UserType,
        operation: Operation,
        queryset: QuerySet,
    ) -> QuerySet:
        """Checking model permissions may query the database, so async callers should
        wrap this in `sync_to_async`.
        """
        from django_ai_sdk.workflows.models import Workflow, WorkflowRun

        if user is None or not bool(user.is_authenticated):
            return queryset.none()
        if queryset.model is Workflow:
            if operation in self.MANAGE and not _has_perm(user, self.MANAGE_PERMISSION):
                return queryset.none()
            return queryset
        if queryset.model is WorkflowRun:
            if operation in self.READ and _has_perm(user, self.VIEW_RUNS_PERMISSION):
                return queryset
            return queryset.filter(user_id=user.pk)
        return queryset


def _has_perm(user: UserType, perm: str) -> bool:
    has_perm = getattr(user, "has_perm", None)
    return has_perm is not None and bool(has_perm(perm))


async def _ahas_perm(user: UserType, perm: str) -> bool:
    ahas_perm = getattr(user, "ahas_perm", None)
    return ahas_perm is not None and bool(await ahas_perm(perm))


def user_pk(user: UserType) -> Any:
    if user is None or not user.is_authenticated:
        return None
    return user.pk


def _owned_by(owner_id: Any, user: UserType) -> bool:
    if owner_id is None or user is None:
        return False
    return str(owner_id) == str(user.pk)


DOMAIN_PERMISSION_DEFAULTS: dict[PermissionDomain, list[str]] = {
    PermissionDomain.AGENT: ["django_ai_sdk.permissions.AgentDefaultPermission"],
    PermissionDomain.THREAD: ["django_ai_sdk.permissions.ThreadDefaultPermission"],
    PermissionDomain.MEMORY: ["django_ai_sdk.permissions.MemoryDefaultPermission"],
    PermissionDomain.INTEGRATIONS: ["django_ai_sdk.permissions.IntegrationDefaultPermission"],
    PermissionDomain.WORKFLOW: ["django_ai_sdk.permissions.WorkflowDefaultPermission"],
    PermissionDomain.OVERSIGHT: ["django_ai_sdk.oversight.permissions.OversightDefaultPermission"],
}


@lru_cache(maxsize=10)
def get_domain_permissions(domain: PermissionDomain) -> list[type[BasePermission]]:
    overrides = resolve_setting("AI_SDK_PERMISSIONS", {}).get(domain.value)
    if overrides is not None:
        return [import_string(p) for p in overrides] if overrides else []
    paths = DOMAIN_PERMISSION_DEFAULTS.get(domain, [])
    return [import_string(p) for p in paths]


class PermissionsMixin:
    """Subclasses must set ``domain`` to a :class:`PermissionDomain` value."""

    domain: PermissionDomain

    @classmethod
    def get_default_permissions(cls) -> list[type[BasePermission]]:
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
        """Check the user against each upper-case frozenset group (READ, WRITE, MANAGE, ...)."""
        groups: dict[str, Operation] = {}
        for perm_cls in cls.get_default_permissions():
            perm = ensure_permission_instance(perm_cls)
            for attr in dir(perm):
                val = getattr(perm, attr, None)
                if attr.isupper() and isinstance(val, frozenset) and val:
                    key = attr.lower()
                    if key not in groups:
                        groups[key] = next(iter(val))
        result: dict[str, bool] = {}
        for name, op in groups.items():
            try:
                await cls.has_perms(user, op, obj, raise_on_deny=True)
                result[name] = True
            except PermissionDenied:
                result[name] = False
        return result

    @classmethod
    async def get_object_permissions(cls, user: UserType, obj: Any) -> ObjectPermissions:
        """The `can_*` flags a client needs to show or hide actions on `obj`."""
        flags = await cls.get_object_permissions_map(user, obj)
        return ObjectPermissions(
            can_read=flags.get("read", False),
            can_write=flags.get("write", False),
            can_manage=flags.get("manage", False),
            can_delete=flags.get("delete", False),
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
        """Filter a queryset through the domain's permission classes; unknown models pass through."""
        perms = permissions if permissions is not None else cls.get_default_permissions()
        for perm_cls in perms:
            perm = ensure_permission_instance(perm_cls)
            queryset = perm.get_queryset_perms(user, operation, queryset)
        return queryset


def ensure_permission_instance(
    perm: type[BasePermission] | BasePermission,
) -> BasePermission:
    return perm() if isinstance(perm, type) else perm


def get_agent_permissions(agent: Agent | None) -> list[type[BasePermission]]:
    perms = getattr(agent, "permissions", None) if agent is not None else None
    if perms:
        return perms
    return get_domain_permissions(PermissionDomain.AGENT)


def get_integration_permissions(service: Any) -> list[type[BasePermission]]:
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
    can_delete: bool = False
