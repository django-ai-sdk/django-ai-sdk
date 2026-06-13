from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from asgiref.sync import async_to_sync, sync_to_async
from django.conf import settings
from django.utils import timezone

from django_ai_sdk.assistants.registry import registry
from django_ai_sdk.permissions import (
    Operation,
    PermissionDenied,
    check_permissions,
    get_default_permissions,
)

if TYPE_CHECKING:
    from typing import Any

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.assistant import Assistant
    from django_ai_sdk.assistants.mixins import AssistantInfo
    from django_ai_sdk.mcp.schemas import AssistantMCPServerStatus


class AssistantSummary(TypedDict):
    id: str
    name: str | None
    model: str | None


class AssistantService:
    """
    Service for resolving assistants from the registry.
    """

    @staticmethod
    def from_registry(assistant_id: str) -> Assistant:
        """Resolve assistant from registry or raise ValueError."""
        assistant = registry.get(assistant_id)
        if assistant is None:
            raise ValueError(f"Assistant '{assistant_id}' not found")
        return assistant

    @staticmethod
    async def get_assistant(
        thread_id: str, user: AbstractBaseUser | AnonymousUser | None = None
    ) -> Assistant:
        """Find the thread and return its associated assistant.

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
        return AssistantService.from_registry(thread.assistant_id)

    @staticmethod
    async def list_assistants(
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> list[AssistantSummary]:
        """Return all registered assistants the user is allowed to view."""
        result: list[AssistantSummary] = []
        for aid, assistant in registry.visible().items():
            perm_classes: list = getattr(assistant, "permissions", get_default_permissions())
            try:
                await check_permissions(user, Operation.VIEW_ASSISTANT, perm_classes)
                result.append(AssistantSummary(id=aid, name=assistant.name, model=assistant.model))
            except PermissionDenied:
                continue
        return result

    @staticmethod
    async def get_assistant_info(
        assistant_id: str, user: AbstractBaseUser | AnonymousUser | None = None
    ) -> AssistantInfo:
        """Return assistant info if user has VIEW_ASSISTANT permission."""
        assistant = AssistantService.from_registry(assistant_id)
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
