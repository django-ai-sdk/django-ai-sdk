from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Required, TypedDict

from asgiref.sync import async_to_sync
from django.core.exceptions import ValidationError

from django_ai_sdk.assistants.registry import registry
from django_ai_sdk.logger import get_logger
from django_ai_sdk.permissions import (
    Operation,
    PermissionDenied,
    PermissionDomain,
    PermissionsMixin,
    get_assistant_permissions,
    has_perms,
)

_logger = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from django_ai_sdk.assistant import Assistant
    from django_ai_sdk.assistants.mixins import AssistantInfo
    from django_ai_sdk.assistants.models import (
        AssistantGroup,
        AssistantUser,
    )
    from django_ai_sdk.integrations.schemas import AssistantIntegrationStatus
    from django_ai_sdk.types import UserType


class AssistantSummary(TypedDict):
    id: str
    name: str | None
    model: str | None
    file_upload: bool
    rag: bool


class AssistantCreateData(TypedDict, total=False):
    name: Required[str]
    slug: str
    assistant: str
    model: str
    system_prompt: str
    tools: list[str]
    integrations: list[str]
    memories: list[str]
    suggestion_enabled: bool
    title_generation: bool
    max_history: int | None
    file_upload: bool


class AssistantUpdateData(TypedDict, total=False):
    name: str
    assistant: str
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


class AssistantService(PermissionsMixin):
    """
    Service for resolving assistants from the registry.
    """

    domain = PermissionDomain.ASSISTANT

    @classmethod
    async def has_perms(
        cls,
        user: UserType,
        operation: Operation,
        obj: Any = None,
        *,
        assistant: Any = None,
        raise_on_deny: bool = True,
        **kwargs: Any,
    ) -> bool:
        perms = get_assistant_permissions(assistant)
        return await has_perms(
            user,
            operation,
            obj,
            permissions=perms,
            raise_on_deny=raise_on_deny,
            **kwargs,
        )

    @classmethod
    def from_registry(cls, assistant_id: str) -> Assistant:
        """Resolve assistant from registry only (sync). Raises ValueError if not found."""
        assistant = registry.get(assistant_id)
        if assistant is None:
            raise ValueError(f"Assistant '{assistant_id}' not found")
        return assistant

    @classmethod
    async def get(cls, assistant_id: str) -> Assistant:
        """Resolve assistant from registry, falling back to AssistantSettings (async)."""
        assistant = registry.get(assistant_id)
        if assistant is not None:
            return assistant
        from django_ai_sdk.assistants.config import get_runtime_assistant_class
        from django_ai_sdk.assistants.models import AssistantSettings

        try:
            config = await AssistantSettings.objects.aget(id=assistant_id, active=True)
        except (AssistantSettings.DoesNotExist, ValidationError) as exc:
            _logger.warning("RuntimeAssistant lookup failed for %s: %r", assistant_id, exc)
            raise ValueError(f"Assistant '{assistant_id}' not found")
        return get_runtime_assistant_class(config.assistant)(config)

    @classmethod
    async def get_assistant(cls, thread_id: str, user: UserType) -> Assistant:
        """Find the thread and return its associated assistant.

        Checks the registry first; falls back to AssistantSettings for
        DB-configured assistants.

        Args:
            thread_id: Thread ID to look up
            user: Optional user for permission checking on the thread lookup

        Returns:
            The assistant instance associated with the thread

        Raises:
            ValueError: If thread or assistant not found
        """
        from django_ai_sdk.storage.services import ThreadService

        thread = await ThreadService.get_thread(thread_id, user=user)
        if thread is None:
            raise ValueError("Thread not found")

        assistant_id = thread.assistant_id
        assistant = registry.get(assistant_id)
        if assistant is not None:
            return assistant

        from django_ai_sdk.assistants.config import get_runtime_assistant_class
        from django_ai_sdk.assistants.models import AssistantSettings

        try:
            config = await AssistantSettings.objects.aget(id=assistant_id, active=True)
        except (AssistantSettings.DoesNotExist, ValidationError) as exc:
            _logger.warning("RuntimeAssistant lookup failed for %s: %r", assistant_id, exc)
            raise ValueError(f"Assistant '{assistant_id}' not found")
        return get_runtime_assistant_class(config.assistant)(config)

    @classmethod
    async def list_assistants(
        cls,
        user: UserType,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AssistantSummary]:
        """Return all assistants the user is allowed to view (registry + DB-backed)."""
        result: list[AssistantSummary] = []

        for aid, assistant in registry.visible().items():
            try:
                await cls.has_perms(user, Operation.VIEW_ASSISTANT, assistant=assistant)
                result.append(
                    AssistantSummary(
                        id=aid,
                        name=assistant.name,
                        model=assistant.model,
                        file_upload=getattr(assistant, "file_upload", False),
                        rag=True if getattr(assistant, "rag_provider", False) else False,
                    )
                )
            except PermissionDenied:
                continue

        from django_ai_sdk.assistants.config import get_runtime_assistant_class
        from django_ai_sdk.assistants.models import AssistantSettings

        async for config in AssistantSettings.objects.filter(active=True):
            try:
                assistant = get_runtime_assistant_class(config.assistant)(config)
            except Exception:
                _logger.exception(
                    "Skipping assistant %r (id=%s): failed to instantiate",
                    config.name,
                    config.id,
                )
                continue
            try:
                await cls.has_perms(user, Operation.VIEW_ASSISTANT, obj=config, assistant=assistant)
            except PermissionDenied:
                continue
            result.append(
                AssistantSummary(
                    id=str(config.id),
                    name=config.name,
                    model=config.model,
                    file_upload=getattr(assistant, "file_upload", False),
                    rag=True if getattr(assistant, "rag_provider", None) else False,
                )
            )

        return result[offset : offset + limit]

    @classmethod
    async def get_assistant_info(cls, assistant_id: str, user: UserType) -> AssistantInfo:
        """Return assistant info if user has VIEW_ASSISTANT permission."""
        assistant = await cls.get(assistant_id)
        await cls.has_perms(user, Operation.VIEW_ASSISTANT, assistant=assistant)
        return assistant.info()

    @classmethod
    async def get_integration_status(
        cls, assistant: Any, *, user: UserType
    ) -> list[AssistantIntegrationStatus]:
        """Get integration status for every integration configured on an assistant.

        Requires VIEW_ASSISTANT permission.

        Returns a list of AssistantIntegrationStatus, each carrying an IntegrationStatus
        (see django_ai_sdk.integrations.base). `type`/`tool_names` come from the
        Integration itself (`.kind`/`.get_tool_names()`).
        """
        await cls.has_perms(user, Operation.VIEW_ASSISTANT, assistant=assistant)

        integration_names: list[str] = list(getattr(assistant, "integrations", []) or [])
        if not integration_names:
            return []

        from django_ai_sdk.integrations.registry import get_integrations
        from django_ai_sdk.integrations.schemas import AssistantIntegrationStatus
        from django_ai_sdk.integrations.services import _safe_status_and_tools

        async def _status_for(name: str, integration: Any) -> AssistantIntegrationStatus:
            status, tool_names = await _safe_status_and_tools(name, integration, user)
            return AssistantIntegrationStatus(
                server_name=name,
                label=integration.label,
                type=integration.kind,
                status=status,
                tool_names=tool_names,
            )

        # Run every integration concurrently — each get_status()/get_tool_names() is
        # individually bounded — mirroring Assistant._get_integration_tools().
        integrations = await get_integrations(integration_names)
        return list(
            await asyncio.gather(*(_status_for(name, i) for name, i in integrations.items()))
        )

    # ============================================================================
    # Assistant user management
    # ============================================================================

    @classmethod
    async def list_assistant_users(
        cls, assistant_id: str, *, user: UserType
    ) -> Sequence[AssistantUser]:
        from django_ai_sdk.assistants.models import AssistantSettings, AssistantUser

        assistant = await cls.get(assistant_id)
        try:
            config = await AssistantSettings.objects.aget(id=assistant_id)
        except AssistantSettings.DoesNotExist:
            raise ValueError(f"Assistant '{assistant_id}' not found")
        await cls.has_perms(user, Operation.UPDATE_ASSISTANT, obj=config, assistant=assistant)

        return [
            u
            async for u in AssistantUser.objects.filter(assistant_id=assistant_id).select_related(
                "user"
            )
        ]

    @classmethod
    async def add_assistant_user(
        cls,
        assistant_id: str,
        target_user_id: Any,
        can_manage: bool = False,
        *,
        user: UserType,
    ) -> AssistantUser:
        from django.contrib.auth import get_user_model

        from django_ai_sdk.assistants.models import AssistantSettings, AssistantUser

        assistant = await cls.get(assistant_id)
        try:
            config = await AssistantSettings.objects.aget(id=assistant_id)
        except AssistantSettings.DoesNotExist:
            raise ValueError(f"Assistant '{assistant_id}' not found")
        await cls.has_perms(user, Operation.UPDATE_ASSISTANT, obj=config, assistant=assistant)

        User = get_user_model()
        try:
            target_user = await User.objects.aget(id=target_user_id)
        except User.DoesNotExist:
            raise ValueError(f"User '{target_user_id}' not found")

        entry, _ = await AssistantUser.objects.aupdate_or_create(
            assistant_id=assistant_id,
            user=target_user,
            defaults={"can_manage": can_manage},
        )
        return entry

    @classmethod
    async def update_assistant_user(
        cls,
        assistant_id: str,
        target_user_id: Any,
        can_manage: bool,
        *,
        user: UserType,
    ) -> AssistantUser:
        from django_ai_sdk.assistants.models import AssistantSettings, AssistantUser

        assistant = await cls.get(assistant_id)
        try:
            config = await AssistantSettings.objects.aget(id=assistant_id)
        except AssistantSettings.DoesNotExist:
            raise ValueError(f"Assistant '{assistant_id}' not found")
        await cls.has_perms(user, Operation.UPDATE_ASSISTANT, obj=config, assistant=assistant)

        try:
            entry = await AssistantUser.objects.select_related("user").aget(
                assistant_id=assistant_id, user_id=target_user_id
            )
        except AssistantUser.DoesNotExist:
            raise ValueError(f"User '{target_user_id}' not found on assistant '{assistant_id}'")

        entry.can_manage = can_manage
        await entry.asave(update_fields=["can_manage"])
        return entry

    @classmethod
    async def remove_assistant_user(
        cls,
        assistant_id: str,
        target_user_id: Any,
        *,
        user: UserType,
    ) -> None:
        from django_ai_sdk.assistants.models import AssistantSettings, AssistantUser

        assistant = await cls.get(assistant_id)
        try:
            config = await AssistantSettings.objects.aget(id=assistant_id)
        except AssistantSettings.DoesNotExist:
            raise ValueError(f"Assistant '{assistant_id}' not found")
        await cls.has_perms(user, Operation.UPDATE_ASSISTANT, obj=config, assistant=assistant)

        deleted, _ = await AssistantUser.objects.filter(
            assistant_id=assistant_id, user_id=target_user_id
        ).adelete()
        if not deleted:
            raise ValueError(f"User '{target_user_id}' not found on assistant '{assistant_id}'")

    # ============================================================================
    # Assistant group management
    # ============================================================================

    @classmethod
    async def list_assistant_groups(
        cls, assistant_id: str, *, user: UserType
    ) -> Sequence[AssistantGroup]:
        from django_ai_sdk.assistants.models import AssistantGroup, AssistantSettings

        assistant = await cls.get(assistant_id)
        try:
            config = await AssistantSettings.objects.aget(id=assistant_id)
        except AssistantSettings.DoesNotExist:
            raise ValueError(f"Assistant '{assistant_id}' not found")
        await cls.has_perms(user, Operation.UPDATE_ASSISTANT, obj=config, assistant=assistant)

        return [
            g
            async for g in AssistantGroup.objects.filter(assistant_id=assistant_id).select_related(
                "group"
            )
        ]

    @classmethod
    async def add_assistant_group(
        cls,
        assistant_id: str,
        group_id: Any,
        can_manage: bool = False,
        *,
        user: UserType,
    ) -> AssistantGroup:
        from django.contrib.auth.models import Group

        from django_ai_sdk.assistants.models import AssistantGroup, AssistantSettings

        assistant = await cls.get(assistant_id)
        try:
            config = await AssistantSettings.objects.aget(id=assistant_id)
        except AssistantSettings.DoesNotExist:
            raise ValueError(f"Assistant '{assistant_id}' not found")
        await cls.has_perms(user, Operation.UPDATE_ASSISTANT, obj=config, assistant=assistant)

        try:
            target_group = await Group.objects.aget(id=group_id)
        except Group.DoesNotExist:
            raise ValueError(f"Group '{group_id}' not found")

        entry, _ = await AssistantGroup.objects.aupdate_or_create(
            assistant_id=assistant_id,
            group=target_group,
            defaults={"can_manage": can_manage},
        )
        return entry

    @classmethod
    async def remove_assistant_group(
        cls,
        assistant_id: str,
        group_id: Any,
        *,
        user: UserType,
    ) -> None:
        from django_ai_sdk.assistants.models import AssistantGroup, AssistantSettings

        assistant = await cls.get(assistant_id)
        try:
            config = await AssistantSettings.objects.aget(id=assistant_id)
        except AssistantSettings.DoesNotExist:
            raise ValueError(f"Assistant '{assistant_id}' not found")
        await cls.has_perms(user, Operation.UPDATE_ASSISTANT, obj=config, assistant=assistant)

        deleted, _ = await AssistantGroup.objects.filter(
            assistant_id=assistant_id, group_id=group_id
        ).adelete()
        if not deleted:
            raise ValueError(f"Group '{group_id}' not found on assistant '{assistant_id}'")

    # ============================================================================
    # Runtime assistant CRUD
    # ============================================================================

    @classmethod
    async def list_runtime_assistants(
        cls,
        user: UserType,
    ) -> list[Any]:
        from django_ai_sdk.assistants.config import get_runtime_assistant_class
        from django_ai_sdk.assistants.models import AssistantSettings

        result: list[Any] = []
        async for config in AssistantSettings.objects.all().order_by("name"):
            try:
                assistant = get_runtime_assistant_class(config.assistant)(config)
            except Exception:
                _logger.exception(
                    "Skipping runtime assistant %r (id=%s): failed to instantiate",
                    config.name,
                    config.id,
                )
                continue
            try:
                await cls.has_perms(user, Operation.VIEW_ASSISTANT, obj=config, assistant=assistant)
            except PermissionDenied:
                continue
            result.append(config)
        return result

    @classmethod
    async def get_runtime_assistant(cls, assistant_id: str, user: UserType) -> Any:
        from django_ai_sdk.assistants.config import get_runtime_assistant_class
        from django_ai_sdk.assistants.models import AssistantSettings

        try:
            config = await AssistantSettings.objects.aget(id=assistant_id)
        except AssistantSettings.DoesNotExist:
            raise ValueError(f"Assistant '{assistant_id}' not found")
        assistant = get_runtime_assistant_class(config.assistant)(config)
        await cls.has_perms(user, Operation.VIEW_ASSISTANT, obj=config, assistant=assistant)
        return config

    @classmethod
    async def create_runtime_assistant(
        cls,
        data: AssistantCreateData,
        user: UserType,
    ) -> Any:
        from django_ai_sdk.assistants.models import AssistantSettings, AssistantUser

        await cls.has_perms(user, Operation.CREATE_ASSISTANT)

        config = AssistantSettings(
            name=data["name"],
            slug=data.get("slug", ""),
            assistant=data.get("assistant", ""),
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
            await AssistantUser.objects.aupdate_or_create(
                assistant=config,
                user=user,
                defaults={"can_manage": True},
            )
        return config

    @classmethod
    async def update_runtime_assistant(
        cls,
        assistant_id: str,
        data: AssistantUpdateData,
        user: UserType,
    ) -> Any:
        from django_ai_sdk.assistants.config import get_runtime_assistant_class
        from django_ai_sdk.assistants.models import AssistantSettings

        try:
            config = await AssistantSettings.objects.aget(id=assistant_id)
        except AssistantSettings.DoesNotExist:
            raise ValueError(f"Assistant '{assistant_id}' not found")
        assistant = get_runtime_assistant_class(config.assistant)(config)
        await cls.has_perms(user, Operation.UPDATE_ASSISTANT, obj=config, assistant=assistant)

        update_fields: list[str] = []
        for field, value in data.items():
            setattr(config, field, value)
            update_fields.append(field)

        if update_fields:
            update_fields.append("updated_at")
            await config.asave(update_fields=update_fields)

        return config

    @classmethod
    async def delete_runtime_assistant(cls, assistant_id: str, user: UserType) -> Any:
        from django_ai_sdk.assistants.config import get_runtime_assistant_class
        from django_ai_sdk.assistants.models import AssistantSettings

        try:
            config = await AssistantSettings.objects.aget(id=assistant_id)
        except AssistantSettings.DoesNotExist:
            raise ValueError(f"Assistant '{assistant_id}' not found")
        assistant = get_runtime_assistant_class(config.assistant)(config)
        await cls.has_perms(user, Operation.DELETE_ASSISTANT, obj=config, assistant=assistant)

        await config.adelete()
        return config


list_assistants = async_to_sync(AssistantService.list_assistants)
get_assistant_info = async_to_sync(AssistantService.get_assistant_info)
get_integration_status = async_to_sync(AssistantService.get_integration_status)
list_assistant_users = async_to_sync(AssistantService.list_assistant_users)
add_assistant_user = async_to_sync(AssistantService.add_assistant_user)
update_assistant_user = async_to_sync(AssistantService.update_assistant_user)
remove_assistant_user = async_to_sync(AssistantService.remove_assistant_user)
list_assistant_groups = async_to_sync(AssistantService.list_assistant_groups)
add_assistant_group = async_to_sync(AssistantService.add_assistant_group)
remove_assistant_group = async_to_sync(AssistantService.remove_assistant_group)
list_runtime_assistants = async_to_sync(AssistantService.list_runtime_assistants)
get_runtime_assistant = async_to_sync(AssistantService.get_runtime_assistant)
create_runtime_assistant = async_to_sync(AssistantService.create_runtime_assistant)
update_runtime_assistant = async_to_sync(AssistantService.update_runtime_assistant)
delete_runtime_assistant = async_to_sync(AssistantService.delete_runtime_assistant)
