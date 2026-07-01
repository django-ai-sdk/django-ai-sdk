from __future__ import annotations

from typing import TYPE_CHECKING, Required, TypedDict

from asgiref.sync import async_to_sync, sync_to_async
from django.conf import settings
from django.core.exceptions import ValidationError
from django.utils import timezone

from django_ai_sdk.assistants.registry import registry
from django_ai_sdk.logger import get_logger
from django_ai_sdk.permissions import (
    Operation,
    PermissionDenied,
    get_assistant_permissions,
    has_perms,
)

_logger = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.assistant import Assistant
    from django_ai_sdk.assistants.mixins import AssistantInfo
    from django_ai_sdk.assistants.models import AssistantGroup, AssistantSettings, AssistantUser
    from django_ai_sdk.mcp.schemas import AssistantMCPServerStatus


class AssistantSummary(TypedDict):
    id: str
    name: str | None
    model: str | None


class AssistantCreateData(TypedDict, total=False):
    name: Required[str]
    slug: str
    assistant: str
    model: str
    system_prompt: str
    tools: list[str]
    mcp_servers: list[str]
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
    mcp_servers: list[str]
    memories: list[str]
    suggestion_enabled: bool
    title_generation: bool
    max_history: int | None
    file_upload: bool
    active: bool


class AssistantService:
    """
    Service for resolving assistants from the registry.
    """

    @staticmethod
    def from_registry(assistant_id: str) -> Assistant:
        """Resolve assistant from registry only (sync). Raises ValueError if not found."""
        assistant = registry.get(assistant_id)
        if assistant is None:
            raise ValueError(f"Assistant '{assistant_id}' not found")
        return assistant

    @staticmethod
    async def get(assistant_id: str) -> Assistant:
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

    @staticmethod
    async def get_assistant(
        thread_id: str, user: AbstractBaseUser | AnonymousUser | None = None
    ) -> Assistant:
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

    @staticmethod
    async def list_assistants(
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> list[AssistantSummary]:
        """Return all assistants the user is allowed to view (registry + DB-backed)."""
        result: list[AssistantSummary] = []

        for aid, assistant in registry.visible().items():
            permissions = get_assistant_permissions(assistant)
            try:
                await has_perms(user, Operation.VIEW_ASSISTANT, permissions)
                result.append(AssistantSummary(id=aid, name=assistant.name, model=assistant.model))
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
            permissions = get_assistant_permissions(assistant)
            try:
                await has_perms(user, Operation.VIEW_ASSISTANT, permissions)
            except PermissionDenied:
                continue
            result.append(AssistantSummary(id=str(config.id), name=config.name, model=config.model))

        return result

    @staticmethod
    async def get_assistant_info(
        assistant_id: str, user: AbstractBaseUser | AnonymousUser | None = None
    ) -> AssistantInfo:
        """Return assistant info if user has VIEW_ASSISTANT permission."""
        assistant = await AssistantService.get(assistant_id)
        await has_perms(user, Operation.VIEW_ASSISTANT, get_assistant_permissions(assistant))
        return assistant.info()

    @staticmethod
    async def get_mcp_server_status(
        assistant: Any, *, user: AbstractBaseUser | AnonymousUser | None = None
    ) -> list[AssistantMCPServerStatus]:
        """Get MCP server connection status for an assistant.

        Requires VIEW_ASSISTANT permission.

        Returns a list of AssistantMCPServerStatus with status: 'active', 'expired', or 'disconnected'.
        """
        await has_perms(user, Operation.VIEW_ASSISTANT, get_assistant_permissions(assistant))

        try:
            from django_ai_sdk.mcp.constants import (
                MCP_STATUS_ACTIVE,
                MCP_STATUS_DISCONNECTED,
                MCP_STATUS_EXPIRED,
            )
            from django_ai_sdk.mcp.models import MCPOAuthToken
            from django_ai_sdk.mcp.schemas import AssistantMCPServerStatus, OAuthMCPServer
        except ImportError:
            return []

        mcp_server_names: list[str] = getattr(assistant, "mcp_servers", [])
        if not mcp_server_names:
            return []

        all_servers = getattr(settings, "AI_SDK_MCP_SERVERS", {})

        def _get_tokens() -> dict[str, dict]:
            return {
                row["server_name"]: row
                for row in MCPOAuthToken.objects.filter(
                    user=user, server_name__in=mcp_server_names
                ).values("server_name", "expires_at")
            }

        oauth_tokens = await sync_to_async(_get_tokens)()

        now = timezone.now()
        result = []
        for name in mcp_server_names:
            server = all_servers.get(name)
            if server is None:
                continue

            if isinstance(server, OAuthMCPServer):
                token_row = oauth_tokens.get(name)
                if token_row is None:
                    status = MCP_STATUS_DISCONNECTED
                elif token_row["expires_at"] and token_row["expires_at"] <= now:
                    status = MCP_STATUS_EXPIRED
                else:
                    status = MCP_STATUS_ACTIVE
            else:
                status = MCP_STATUS_ACTIVE

            result.append(
                AssistantMCPServerStatus(
                    server_name=name,
                    label=server.label or name.title(),
                    type=server.type,
                    status=status,
                    tool_names=server.tools or [],
                )
            )

        return result

    # ============================================================================
    # Assistant user management
    # ============================================================================

    @staticmethod
    async def list_assistant_users(
        assistant_id: str, *, user: AbstractBaseUser | AnonymousUser | None = None
    ) -> Sequence[AssistantUser]:
        from django_ai_sdk.assistants.models import AssistantUser

        return [
            u
            async for u in AssistantUser.objects.filter(assistant_id=assistant_id).select_related(
                "user"
            )
        ]

    @staticmethod
    async def add_assistant_user(
        assistant_id: str,
        target_user_id: Any,
        can_manage: bool = False,
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> AssistantUser:
        from django.contrib.auth import get_user_model

        from django_ai_sdk.assistants.models import AssistantSettings, AssistantUser

        assistant_obj = await AssistantService.get(assistant_id)
        permissions = get_assistant_permissions(assistant_obj)
        try:
            config = await AssistantSettings.objects.aget(id=assistant_id)
        except AssistantSettings.DoesNotExist:
            raise ValueError(f"Assistant '{assistant_id}' not found")
        await has_perms(user, Operation.UPDATE_ASSISTANT, config, permissions=permissions)

        User = get_user_model()
        try:
            target_user = await User.objects.aget(id=target_user_id)
        except User.DoesNotExist:
            raise ValueError(f"User '{target_user_id}' not found")

        entry, _ = await AssistantUser.objects.aget_or_create(
            assistant_id=assistant_id,
            user=target_user,
            defaults={"can_manage": can_manage},
        )
        return entry

    @staticmethod
    async def update_assistant_user(
        assistant_id: str,
        target_user_id: Any,
        can_manage: bool,
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> AssistantUser:
        from django_ai_sdk.assistants.models import AssistantSettings, AssistantUser

        assistant_obj = await AssistantService.get(assistant_id)
        permissions = get_assistant_permissions(assistant_obj)
        try:
            config = await AssistantSettings.objects.aget(id=assistant_id)
        except AssistantSettings.DoesNotExist:
            raise ValueError(f"Assistant '{assistant_id}' not found")
        await has_perms(user, Operation.UPDATE_ASSISTANT, config, permissions=permissions)

        try:
            entry = await AssistantUser.objects.select_related("user").aget(
                assistant_id=assistant_id, user_id=target_user_id
            )
        except AssistantUser.DoesNotExist:
            raise ValueError(f"User '{target_user_id}' not found on assistant '{assistant_id}'")

        entry.can_manage = can_manage
        await entry.asave(update_fields=["can_manage"])
        return entry

    @staticmethod
    async def remove_assistant_user(
        assistant_id: str,
        target_user_id: Any,
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> None:
        from django_ai_sdk.assistants.models import AssistantSettings, AssistantUser

        assistant_obj = await AssistantService.get(assistant_id)
        permissions = get_assistant_permissions(assistant_obj)
        try:
            config = await AssistantSettings.objects.aget(id=assistant_id)
        except AssistantSettings.DoesNotExist:
            raise ValueError(f"Assistant '{assistant_id}' not found")
        await has_perms(user, Operation.UPDATE_ASSISTANT, config, permissions=permissions)

        deleted, _ = await AssistantUser.objects.filter(
            assistant_id=assistant_id, user_id=target_user_id
        ).adelete()
        if not deleted:
            raise ValueError(f"User '{target_user_id}' not found on assistant '{assistant_id}'")

    # ============================================================================
    # Assistant group management
    # ============================================================================

    @staticmethod
    async def list_assistant_groups(
        assistant_id: str, *, user: AbstractBaseUser | AnonymousUser | None = None
    ) -> Sequence[AssistantGroup]:
        from django_ai_sdk.assistants.models import AssistantGroup

        return [
            g
            async for g in AssistantGroup.objects.filter(assistant_id=assistant_id).select_related(
                "group"
            )
        ]

    @staticmethod
    async def add_assistant_group(
        assistant_id: str,
        group_id: Any,
        can_manage: bool = False,
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> AssistantGroup:
        from django.contrib.auth.models import Group

        from django_ai_sdk.assistants.models import AssistantGroup, AssistantSettings

        assistant_obj = await AssistantService.get(assistant_id)
        permissions = get_assistant_permissions(assistant_obj)
        try:
            config = await AssistantSettings.objects.aget(id=assistant_id)
        except AssistantSettings.DoesNotExist:
            raise ValueError(f"Assistant '{assistant_id}' not found")
        await has_perms(user, Operation.UPDATE_ASSISTANT, config, permissions=permissions)

        try:
            target_group = await Group.objects.aget(id=group_id)
        except Group.DoesNotExist:
            raise ValueError(f"Group '{group_id}' not found")

        entry, _ = await AssistantGroup.objects.aget_or_create(
            assistant_id=assistant_id,
            group=target_group,
            defaults={"can_manage": can_manage},
        )
        return entry

    @staticmethod
    async def remove_assistant_group(
        assistant_id: str,
        group_id: Any,
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> None:
        from django_ai_sdk.assistants.models import AssistantGroup, AssistantSettings

        assistant_obj = await AssistantService.get(assistant_id)
        permissions = get_assistant_permissions(assistant_obj)
        try:
            config = await AssistantSettings.objects.aget(id=assistant_id)
        except AssistantSettings.DoesNotExist:
            raise ValueError(f"Assistant '{assistant_id}' not found")
        await has_perms(user, Operation.UPDATE_ASSISTANT, config, permissions=permissions)

        deleted, _ = await AssistantGroup.objects.filter(
            assistant_id=assistant_id, group_id=group_id
        ).adelete()
        if not deleted:
            raise ValueError(f"Group '{group_id}' not found on assistant '{assistant_id}'")


list_assistants = async_to_sync(AssistantService.list_assistants)
get_assistant_info = async_to_sync(AssistantService.get_assistant_info)
get_mcp_server_status = async_to_sync(AssistantService.get_mcp_server_status)
list_assistant_users = async_to_sync(AssistantService.list_assistant_users)
add_assistant_user = async_to_sync(AssistantService.add_assistant_user)
update_assistant_user = async_to_sync(AssistantService.update_assistant_user)
remove_assistant_user = async_to_sync(AssistantService.remove_assistant_user)
list_assistant_groups = async_to_sync(AssistantService.list_assistant_groups)
add_assistant_group = async_to_sync(AssistantService.add_assistant_group)
remove_assistant_group = async_to_sync(AssistantService.remove_assistant_group)


class AssistantSettingsService:
    """CRUD service for AssistantSettings — shared between DRF and Ninja views."""

    @staticmethod
    async def all() -> list[Any]:
        from django_ai_sdk.assistants.models import AssistantSettings

        return [config async for config in AssistantSettings.objects.all().order_by("name")]

    @staticmethod
    async def get(assistant_id: str) -> AssistantSettings:
        from django_ai_sdk.assistants.models import AssistantSettings

        try:
            return await AssistantSettings.objects.aget(id=assistant_id)
        except AssistantSettings.DoesNotExist:
            raise ValueError(f"Assistant '{assistant_id}' not found")

    @staticmethod
    async def create(
        data: AssistantCreateData,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> AssistantSettings:
        from django_ai_sdk.assistants.models import AssistantSettings

        config = AssistantSettings(
            name=data["name"],
            slug=data.get("slug", ""),
            assistant=data.get("assistant", ""),
            model=data.get("model", "gpt-4o"),
            system_prompt=data.get("system_prompt", ""),
            tools=data.get("tools", []),
            mcp_servers=data.get("mcp_servers", []),
            memories=data.get("memories", []),
            suggestion_enabled=data.get("suggestion_enabled", False),
            title_generation=data.get("title_generation", True),
            max_history=data.get("max_history"),
            file_upload=data.get("file_upload", False),
        )
        await config.asave()
        return config

    @staticmethod
    async def update(
        assistant_id: str,
        data: AssistantUpdateData,
    ) -> AssistantSettings:
        from django_ai_sdk.assistants.models import AssistantSettings

        try:
            config = await AssistantSettings.objects.aget(id=assistant_id)
        except AssistantSettings.DoesNotExist:
            raise ValueError(f"Assistant '{assistant_id}' not found")

        update_fields: list[str] = []
        for field, value in data.items():
            setattr(config, field, value)
            update_fields.append(field)

        if update_fields:
            update_fields.append("updated_at")
            await config.asave(update_fields=update_fields)

        return config

    @staticmethod
    async def delete(assistant_id: str) -> AssistantSettings:
        from django_ai_sdk.assistants.models import AssistantSettings

        try:
            config = await AssistantSettings.objects.aget(id=assistant_id)
        except AssistantSettings.DoesNotExist:
            raise ValueError(f"Assistant '{assistant_id}' not found")

        await config.adelete()
        return config
