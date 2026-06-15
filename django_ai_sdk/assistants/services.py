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
    check_permissions,
    get_default_permissions,
)

_logger = get_logger(__name__)

if TYPE_CHECKING:
    from typing import Any

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.assistant import Assistant
    from django_ai_sdk.assistants.mixins import AssistantInfo
    from django_ai_sdk.mcp.schemas import AssistantMCPServerStatus
    from django_ai_sdk.web_assistant.models import WebAssistantSettings


class AssistantSummary(TypedDict):
    id: str
    name: str | None
    model: str | None


class WebAssistantCreateData(TypedDict, total=False):
    name: Required[str]
    slug: str
    base_class: str
    model: str
    system_prompt: str
    tools: list[str]
    mcp_servers: list[str]
    suggestion_enabled: bool
    title_generation: bool
    max_history: int | None
    file_upload: bool


class WebAssistantUpdateData(TypedDict, total=False):
    name: str
    base_class: str
    model: str
    system_prompt: str
    tools: list[str]
    mcp_servers: list[str]
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
        """Resolve assistant from registry, falling back to WebAssistantSettings (async)."""
        assistant = registry.get(assistant_id)
        if assistant is not None:
            return assistant
        from django_ai_sdk.web_assistant.config import get_web_assistant_class
        from django_ai_sdk.web_assistant.models import WebAssistantSettings

        try:
            config = await WebAssistantSettings.objects.aget(id=assistant_id, active=True)
        except (WebAssistantSettings.DoesNotExist, ValidationError) as exc:
            _logger.warning("WebAssistant lookup failed for %s: %r", assistant_id, exc)
            raise ValueError(f"Assistant '{assistant_id}' not found")
        return get_web_assistant_class(config.base_class)(config)

    @staticmethod
    async def get_assistant(
        thread_id: str, user: AbstractBaseUser | AnonymousUser | None = None
    ) -> Assistant:
        """Find the thread and return its associated assistant.

        Checks the registry first; falls back to WebAssistantSettings for
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

        from django_ai_sdk.web_assistant.config import get_web_assistant_class
        from django_ai_sdk.web_assistant.models import WebAssistantSettings

        try:
            config = await WebAssistantSettings.objects.aget(id=assistant_id, active=True)
        except (WebAssistantSettings.DoesNotExist, ValidationError) as exc:
            _logger.warning("WebAssistant lookup failed for %s: %r", assistant_id, exc)
            raise ValueError(f"Assistant '{assistant_id}' not found")
        return get_web_assistant_class(config.base_class)(config)

    @staticmethod
    async def list_assistants(
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> list[AssistantSummary]:
        """Return all assistants the user is allowed to view (registry + DB-backed)."""
        result: list[AssistantSummary] = []

        for aid, assistant in registry.visible().items():
            perm_classes: list = getattr(assistant, "permissions", get_default_permissions())
            try:
                await check_permissions(user, Operation.VIEW_ASSISTANT, perm_classes)
                result.append(AssistantSummary(id=aid, name=assistant.name, model=assistant.model))
            except PermissionDenied:
                continue

        from django_ai_sdk.web_assistant.models import WebAssistantSettings

        async for config in WebAssistantSettings.objects.filter(active=True):
            result.append(AssistantSummary(id=str(config.id), name=config.name, model=config.model))

        return result

    @staticmethod
    async def get_assistant_info(
        assistant_id: str, user: AbstractBaseUser | AnonymousUser | None = None
    ) -> AssistantInfo:
        """Return assistant info if user has VIEW_ASSISTANT permission."""
        assistant = await AssistantService.get(assistant_id)
        perm_classes: list = getattr(assistant, "permissions", get_default_permissions())
        await check_permissions(user, Operation.VIEW_ASSISTANT, perm_classes)
        return assistant.info()

    @staticmethod
    async def get_mcp_server_status(
        assistant: Any, *, user: AbstractBaseUser | AnonymousUser | None = None
    ) -> list[AssistantMCPServerStatus]:
        """Get MCP server connection status for an assistant.

        Requires VIEW_ASSISTANT permission.

        Returns a list of AssistantMCPServerStatus with status: 'active', 'expired', or 'disconnected'.
        """
        perm_classes: list = getattr(assistant, "permissions", get_default_permissions())
        await check_permissions(user, Operation.VIEW_ASSISTANT, perm_classes)

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


list_assistants = async_to_sync(AssistantService.list_assistants)
get_assistant_info = async_to_sync(AssistantService.get_assistant_info)
get_mcp_server_status = async_to_sync(AssistantService.get_mcp_server_status)


class WebAssistantService:
    """CRUD service for WebAssistantSettings — shared between DRF and Ninja views."""

    @staticmethod
    async def all() -> list[Any]:
        from django_ai_sdk.web_assistant.models import WebAssistantSettings

        return [config async for config in WebAssistantSettings.objects.all().order_by("name")]

    @staticmethod
    async def get(assistant_id: str) -> WebAssistantSettings:
        from django_ai_sdk.web_assistant.models import WebAssistantSettings

        try:
            return await WebAssistantSettings.objects.aget(id=assistant_id)
        except WebAssistantSettings.DoesNotExist:
            raise ValueError(f"Web assistant '{assistant_id}' not found")

    @staticmethod
    async def create(
        data: WebAssistantCreateData,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> WebAssistantSettings:
        from django_ai_sdk.web_assistant.models import WebAssistantSettings

        config = WebAssistantSettings(
            name=data["name"],
            slug=data.get("slug", ""),
            base_class=data.get("base_class", ""),
            model=data.get("model", "gpt-4o"),
            system_prompt=data.get("system_prompt", ""),
            tools=data.get("tools", []),
            mcp_servers=data.get("mcp_servers", []),
            suggestion_enabled=data.get("suggestion_enabled", False),
            title_generation=data.get("title_generation", True),
            max_history=data.get("max_history"),
            file_upload=data.get("file_upload", False),
            created_by=user if getattr(user, "is_authenticated", False) else None,
        )
        await config.asave()
        return config

    @staticmethod
    async def update(
        assistant_id: str,
        data: WebAssistantUpdateData,
    ) -> WebAssistantSettings:
        from django_ai_sdk.web_assistant.models import WebAssistantSettings

        try:
            config = await WebAssistantSettings.objects.aget(id=assistant_id)
        except WebAssistantSettings.DoesNotExist:
            raise ValueError(f"Web assistant '{assistant_id}' not found")

        update_fields: list[str] = []
        for field, value in data.items():
            setattr(config, field, value)
            update_fields.append(field)

        if update_fields:
            update_fields.append("updated_at")
            await config.asave(update_fields=update_fields)

        return config

    @staticmethod
    async def delete(assistant_id: str) -> WebAssistantSettings:
        from django_ai_sdk.web_assistant.models import WebAssistantSettings

        try:
            config = await WebAssistantSettings.objects.aget(id=assistant_id)
        except WebAssistantSettings.DoesNotExist:
            raise ValueError(f"Web assistant '{assistant_id}' not found")

        await config.adelete()
        return config
