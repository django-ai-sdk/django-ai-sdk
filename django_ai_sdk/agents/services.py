from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Required, TypedDict

from asgiref.sync import async_to_sync
from django.core.exceptions import ValidationError

from django_ai_sdk.agents.registry import registry
from django_ai_sdk.logger import get_logger
from django_ai_sdk.permissions import (
    Operation,
    PermissionDenied,
    PermissionDomain,
    PermissionsMixin,
    get_agent_permissions,
    has_perms,
)

_logger = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from django_ai_sdk.agent import Agent
    from django_ai_sdk.agents.mixins import AgentInfo
    from django_ai_sdk.agents.models import (
        AgentGroup,
        AgentUser,
    )
    from django_ai_sdk.integrations.schemas import AgentIntegrationStatus
    from django_ai_sdk.types import UserType


class AgentSummary(TypedDict):
    id: str
    name: str | None
    model: str | None
    file_upload: bool
    rag: bool


class AgentCreateData(TypedDict, total=False):
    name: Required[str]
    slug: str
    agent: str
    model: str
    system_prompt: str
    tools: list[str]
    integrations: list[str]
    memories: list[str]
    suggestion_enabled: bool
    title_generation: bool
    max_history: int | None
    file_upload: bool


class AgentUpdateData(TypedDict, total=False):
    name: str
    agent: str
    model: str
    system_prompt: str
    tools: list[str]
    integrations: list[str]
    memories: list[str]
    suggestion_enabled: bool
    title_generation: bool
    max_history: int | None
    file_upload: bool
    active: bool


class AgentService(PermissionsMixin):
    """
    Service for resolving agents from the registry.
    """

    domain = PermissionDomain.AGENT

    @classmethod
    async def has_perms(
        cls,
        user: UserType,
        operation: Operation,
        obj: Any = None,
        *,
        agent: Any = None,
        raise_on_deny: bool = True,
        **kwargs: Any,
    ) -> bool:
        perms = get_agent_permissions(agent)
        return await has_perms(
            user,
            operation,
            obj,
            permissions=perms,
            raise_on_deny=raise_on_deny,
            **kwargs,
        )

    @classmethod
    def from_registry(cls, agent_id: str) -> Agent:
        """Resolve agent from registry only (sync). Raises ValueError if not found."""
        agent = registry.get(agent_id)
        if agent is None:
            raise ValueError(f"Agent '{agent_id}' not found")
        return agent

    @classmethod
    async def get(cls, agent_id: str) -> Agent:
        """Resolve agent from registry, falling back to AgentSettings (async)."""
        agent = registry.get(agent_id)
        if agent is not None:
            return agent
        from django_ai_sdk.agents.config import get_runtime_agent_class
        from django_ai_sdk.agents.models import AgentSettings

        try:
            config = await AgentSettings.objects.aget(id=agent_id, active=True)
        except (AgentSettings.DoesNotExist, ValidationError) as exc:
            _logger.warning("RuntimeAgent lookup failed for %s: %r", agent_id, exc)
            raise ValueError(f"Agent '{agent_id}' not found")
        return get_runtime_agent_class(config.agent)(config)

    @classmethod
    async def get_agent(cls, thread_id: str, user: UserType) -> Agent:
        """Find the thread and return its associated agent.

        Checks the registry first; falls back to AgentSettings for
        DB-configured agents.

        Args:
            thread_id: Thread ID to look up
            user: Optional user for permission checking on the thread lookup

        Returns:
            The agent instance associated with the thread

        Raises:
            ValueError: If thread or agent not found
        """
        from django_ai_sdk.storage.services import ThreadService

        thread = await ThreadService.get_thread(thread_id, user=user)
        if thread is None:
            raise ValueError("Thread not found")

        agent_id = thread.agent_id
        agent = registry.get(agent_id)
        if agent is not None:
            return agent

        from django_ai_sdk.agents.config import get_runtime_agent_class
        from django_ai_sdk.agents.models import AgentSettings

        try:
            config = await AgentSettings.objects.aget(id=agent_id, active=True)
        except (AgentSettings.DoesNotExist, ValidationError) as exc:
            _logger.warning("RuntimeAgent lookup failed for %s: %r", agent_id, exc)
            raise ValueError(f"Agent '{agent_id}' not found")
        return get_runtime_agent_class(config.agent)(config)

    @classmethod
    async def list_agents(
        cls,
        user: UserType,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AgentSummary]:
        """Return all agents the user is allowed to view (registry + DB-backed)."""
        result: list[AgentSummary] = []

        for aid, agent in registry.visible().items():
            try:
                await cls.has_perms(user, Operation.VIEW_AGENT, agent=agent)
                result.append(
                    AgentSummary(
                        id=aid,
                        name=agent.name,
                        model=agent.model,
                        file_upload=getattr(agent, "file_upload", False),
                        rag=True if getattr(agent, "rag_provider", False) else False,
                    )
                )
            except PermissionDenied:
                continue

        from django_ai_sdk.agents.config import get_runtime_agent_class
        from django_ai_sdk.agents.models import AgentSettings

        async for config in AgentSettings.objects.filter(active=True):
            try:
                agent = get_runtime_agent_class(config.agent)(config)
            except Exception:
                _logger.exception(
                    "Skipping agent %r (id=%s): failed to instantiate",
                    config.name,
                    config.id,
                )
                continue
            try:
                await cls.has_perms(user, Operation.VIEW_AGENT, obj=config, agent=agent)
            except PermissionDenied:
                continue
            result.append(
                AgentSummary(
                    id=str(config.id),
                    name=config.name,
                    model=config.model,
                    file_upload=getattr(agent, "file_upload", False),
                    rag=True if getattr(agent, "rag_provider", None) else False,
                )
            )

        return result[offset : offset + limit]

    @classmethod
    async def get_agent_info(cls, agent_id: str, user: UserType) -> AgentInfo:
        """Return agent info if user has VIEW_AGENT permission."""
        agent = await cls.get(agent_id)
        await cls.has_perms(
            user, Operation.VIEW_AGENT, obj=agent.config if agent.is_runtime else None, agent=agent
        )
        return agent.info()

    @classmethod
    async def get_integration_status(
        cls, agent: Any, *, user: UserType
    ) -> list[AgentIntegrationStatus]:
        """Get integration status for every integration configured on an agent.

        Requires VIEW_AGENT permission.

        Returns a list of AgentIntegrationStatus, each carrying an IntegrationStatus
        (see django_ai_sdk.integrations.base). `type`/`tool_names` come from the
        Integration itself (`.kind`/`.get_tool_names()`).
        """
        await cls.has_perms(
            user, Operation.VIEW_AGENT, obj=agent.config if agent.is_runtime else None, agent=agent
        )

        integration_names: list[str] = list(getattr(agent, "integrations", []) or [])
        if not integration_names:
            return []

        from django_ai_sdk.integrations.registry import get_integrations
        from django_ai_sdk.integrations.schemas import AgentIntegrationStatus
        from django_ai_sdk.integrations.services import _safe_status_and_tools

        async def _status_for(name: str, integration: Any) -> AgentIntegrationStatus:
            status, tool_names = await _safe_status_and_tools(name, integration, user)
            return AgentIntegrationStatus(
                server_name=name,
                label=integration.label,
                type=integration.kind,
                status=status,
                tool_names=tool_names,
            )

        # Run every integration concurrently — each get_status()/get_tool_names() is
        # individually bounded — mirroring Agent._get_integration_tools().
        integrations = await get_integrations(integration_names)
        return list(
            await asyncio.gather(*(_status_for(name, i) for name, i in integrations.items()))
        )

    # ============================================================================
    # Agent user management
    # ============================================================================

    @classmethod
    async def list_agent_users(cls, agent_id: str, *, user: UserType) -> Sequence[AgentUser]:
        from django_ai_sdk.agents.models import AgentSettings, AgentUser

        agent = await cls.get(agent_id)
        try:
            config = await AgentSettings.objects.aget(id=agent_id)
        except AgentSettings.DoesNotExist:
            raise ValueError(f"Agent '{agent_id}' not found")
        await cls.has_perms(user, Operation.UPDATE_AGENT, obj=config, agent=agent)

        return [u async for u in AgentUser.objects.filter(agent_id=agent_id).select_related("user")]

    @classmethod
    async def add_agent_user(
        cls,
        agent_id: str,
        target_user_id: Any,
        can_manage: bool = False,
        *,
        user: UserType,
    ) -> AgentUser:
        from django.contrib.auth import get_user_model

        from django_ai_sdk.agents.models import AgentSettings, AgentUser

        agent = await cls.get(agent_id)
        try:
            config = await AgentSettings.objects.aget(id=agent_id)
        except AgentSettings.DoesNotExist:
            raise ValueError(f"Agent '{agent_id}' not found")
        await cls.has_perms(user, Operation.UPDATE_AGENT, obj=config, agent=agent)

        User = get_user_model()
        try:
            target_user = await User.objects.aget(id=target_user_id)
        except User.DoesNotExist:
            raise ValueError(f"User '{target_user_id}' not found")

        entry, _ = await AgentUser.objects.aupdate_or_create(
            agent_id=agent_id,
            user=target_user,
            defaults={"can_manage": can_manage},
        )
        return entry

    @classmethod
    async def update_agent_user(
        cls,
        agent_id: str,
        target_user_id: Any,
        can_manage: bool,
        *,
        user: UserType,
    ) -> AgentUser:
        from django_ai_sdk.agents.models import AgentSettings, AgentUser

        agent = await cls.get(agent_id)
        try:
            config = await AgentSettings.objects.aget(id=agent_id)
        except AgentSettings.DoesNotExist:
            raise ValueError(f"Agent '{agent_id}' not found")
        await cls.has_perms(user, Operation.UPDATE_AGENT, obj=config, agent=agent)

        try:
            entry = await AgentUser.objects.select_related("user").aget(
                agent_id=agent_id, user_id=target_user_id
            )
        except AgentUser.DoesNotExist:
            raise ValueError(f"User '{target_user_id}' not found on agent '{agent_id}'")

        entry.can_manage = can_manage
        await entry.asave(update_fields=["can_manage"])
        return entry

    @classmethod
    async def remove_agent_user(
        cls,
        agent_id: str,
        target_user_id: Any,
        *,
        user: UserType,
    ) -> None:
        from django_ai_sdk.agents.models import AgentSettings, AgentUser

        agent = await cls.get(agent_id)
        try:
            config = await AgentSettings.objects.aget(id=agent_id)
        except AgentSettings.DoesNotExist:
            raise ValueError(f"Agent '{agent_id}' not found")
        await cls.has_perms(user, Operation.UPDATE_AGENT, obj=config, agent=agent)

        deleted, _ = await AgentUser.objects.filter(
            agent_id=agent_id, user_id=target_user_id
        ).adelete()
        if not deleted:
            raise ValueError(f"User '{target_user_id}' not found on agent '{agent_id}'")

    # ============================================================================
    # Agent group management
    # ============================================================================

    @classmethod
    async def list_agent_groups(cls, agent_id: str, *, user: UserType) -> Sequence[AgentGroup]:
        from django_ai_sdk.agents.models import AgentGroup, AgentSettings

        agent = await cls.get(agent_id)
        try:
            config = await AgentSettings.objects.aget(id=agent_id)
        except AgentSettings.DoesNotExist:
            raise ValueError(f"Agent '{agent_id}' not found")
        await cls.has_perms(user, Operation.UPDATE_AGENT, obj=config, agent=agent)

        return [
            g async for g in AgentGroup.objects.filter(agent_id=agent_id).select_related("group")
        ]

    @classmethod
    async def add_agent_group(
        cls,
        agent_id: str,
        group_id: Any,
        can_manage: bool = False,
        *,
        user: UserType,
    ) -> AgentGroup:
        from django.contrib.auth.models import Group

        from django_ai_sdk.agents.models import AgentGroup, AgentSettings

        agent = await cls.get(agent_id)
        try:
            config = await AgentSettings.objects.aget(id=agent_id)
        except AgentSettings.DoesNotExist:
            raise ValueError(f"Agent '{agent_id}' not found")
        await cls.has_perms(user, Operation.UPDATE_AGENT, obj=config, agent=agent)

        try:
            target_group = await Group.objects.aget(id=group_id)
        except Group.DoesNotExist:
            raise ValueError(f"Group '{group_id}' not found")

        entry, _ = await AgentGroup.objects.aupdate_or_create(
            agent_id=agent_id,
            group=target_group,
            defaults={"can_manage": can_manage},
        )
        return entry

    @classmethod
    async def remove_agent_group(
        cls,
        agent_id: str,
        group_id: Any,
        *,
        user: UserType,
    ) -> None:
        from django_ai_sdk.agents.models import AgentGroup, AgentSettings

        agent = await cls.get(agent_id)
        try:
            config = await AgentSettings.objects.aget(id=agent_id)
        except AgentSettings.DoesNotExist:
            raise ValueError(f"Agent '{agent_id}' not found")
        await cls.has_perms(user, Operation.UPDATE_AGENT, obj=config, agent=agent)

        deleted, _ = await AgentGroup.objects.filter(agent_id=agent_id, group_id=group_id).adelete()
        if not deleted:
            raise ValueError(f"Group '{group_id}' not found on agent '{agent_id}'")

    # ============================================================================
    # Runtime agent CRUD
    # ============================================================================

    @classmethod
    async def list_runtime_agents(
        cls,
        user: UserType,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Any]:
        from django_ai_sdk.agents.config import get_runtime_agent_class
        from django_ai_sdk.agents.models import AgentSettings

        result: list[Any] = []
        async for config in AgentSettings.objects.all().order_by("name"):
            try:
                agent = get_runtime_agent_class(config.agent)(config)
            except Exception:
                _logger.exception(
                    "Skipping runtime agent %r (id=%s): failed to instantiate",
                    config.name,
                    config.id,
                )
                continue
            try:
                await cls.has_perms(user, Operation.VIEW_AGENT, obj=config, agent=agent)
            except PermissionDenied:
                continue
            result.append(config)
        return result[offset : offset + limit]

    @classmethod
    async def get_runtime_agent(cls, agent_id: str, user: UserType) -> Any:
        from django_ai_sdk.agents.config import get_runtime_agent_class
        from django_ai_sdk.agents.models import AgentSettings

        try:
            config = await AgentSettings.objects.aget(id=agent_id)
        except AgentSettings.DoesNotExist:
            raise ValueError(f"Agent '{agent_id}' not found")
        agent = get_runtime_agent_class(config.agent)(config)
        await cls.has_perms(user, Operation.VIEW_AGENT, obj=config, agent=agent)
        return config

    @classmethod
    async def create_runtime_agent(
        cls,
        data: AgentCreateData,
        user: UserType,
    ) -> Any:
        from django_ai_sdk.agents.models import AgentSettings, AgentUser

        await cls.has_perms(user, Operation.CREATE_AGENT)

        config = AgentSettings(
            name=data["name"],
            slug=data.get("slug", ""),
            agent=data.get("agent", ""),
            model=data.get("model", "gpt-4o"),
            system_prompt=data.get("system_prompt", ""),
            tools=data.get("tools", []),
            integrations=data.get("integrations", []),
            memories=data.get("memories", []),
            suggestion_enabled=data.get("suggestion_enabled", False),
            title_generation=data.get("title_generation", True),
            max_history=data.get("max_history"),
            file_upload=data.get("file_upload", False),
        )
        await config.asave()
        if user is not None and bool(user.is_authenticated):
            await AgentUser.objects.aupdate_or_create(
                agent=config,
                user=user,
                defaults={"can_manage": True},
            )
        return config

    @classmethod
    async def update_runtime_agent(
        cls,
        agent_id: str,
        data: AgentUpdateData,
        user: UserType,
    ) -> Any:
        from django_ai_sdk.agents.config import get_runtime_agent_class
        from django_ai_sdk.agents.models import AgentSettings

        try:
            config = await AgentSettings.objects.aget(id=agent_id)
        except AgentSettings.DoesNotExist:
            raise ValueError(f"Agent '{agent_id}' not found")
        agent = get_runtime_agent_class(config.agent)(config)
        await cls.has_perms(user, Operation.UPDATE_AGENT, obj=config, agent=agent)

        update_fields: list[str] = []
        for field, value in data.items():
            setattr(config, field, value)
            update_fields.append(field)

        if update_fields:
            update_fields.append("updated_at")
            await config.asave(update_fields=update_fields)

        return config

    @classmethod
    async def delete_runtime_agent(cls, agent_id: str, user: UserType) -> Any:
        from django_ai_sdk.agents.config import get_runtime_agent_class
        from django_ai_sdk.agents.models import AgentSettings

        try:
            config = await AgentSettings.objects.aget(id=agent_id)
        except AgentSettings.DoesNotExist:
            raise ValueError(f"Agent '{agent_id}' not found")
        agent = get_runtime_agent_class(config.agent)(config)
        await cls.has_perms(user, Operation.DELETE_AGENT, obj=config, agent=agent)

        await config.adelete()
        return config


list_agents = async_to_sync(AgentService.list_agents)
get_agent_info = async_to_sync(AgentService.get_agent_info)
get_integration_status = async_to_sync(AgentService.get_integration_status)
list_agent_users = async_to_sync(AgentService.list_agent_users)
add_agent_user = async_to_sync(AgentService.add_agent_user)
update_agent_user = async_to_sync(AgentService.update_agent_user)
remove_agent_user = async_to_sync(AgentService.remove_agent_user)
list_agent_groups = async_to_sync(AgentService.list_agent_groups)
add_agent_group = async_to_sync(AgentService.add_agent_group)
remove_agent_group = async_to_sync(AgentService.remove_agent_group)
list_runtime_agents = async_to_sync(AgentService.list_runtime_agents)
get_runtime_agent = async_to_sync(AgentService.get_runtime_agent)
create_runtime_agent = async_to_sync(AgentService.create_runtime_agent)
update_runtime_agent = async_to_sync(AgentService.update_runtime_agent)
delete_runtime_agent = async_to_sync(AgentService.delete_runtime_agent)
