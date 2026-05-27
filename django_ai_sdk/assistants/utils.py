"""Utilities for Assistant tool management."""

from typing import Any

from django.conf import settings
from django.utils import timezone

MCP_STATUS_ACTIVE = "active"
MCP_STATUS_EXPIRED = "expired"
MCP_STATUS_DISCONNECTED = "disconnected"
MCP_STATUSES = (MCP_STATUS_ACTIVE, MCP_STATUS_EXPIRED, MCP_STATUS_DISCONNECTED)


async def get_mcp_server_status(assistant: Any, user_id: Any) -> list[dict]:
    """Get MCP server connection status for an assistant.

    Returns a list of dicts with keys: server_name, label, type, status, tool_names, connect_url.
    Status is 'active', 'expired', or 'disconnected'.
    """
    try:
        from django_ai_sdk.mcp.models import MCPOAuthToken
        from django_ai_sdk.mcp.schemas import OAuthMCPServer
    except ImportError:
        return []

    mcp_server_names: list[str] = getattr(assistant, "mcp_servers", [])
    if not mcp_server_names:
        return []

    all_servers = getattr(settings, "AI_SDK_MCP_SERVERS", {})

    oauth_tokens = {
        row["server_name"]: row
        for row in MCPOAuthToken.objects.filter(
            user=user_id, server_name__in=mcp_server_names
        ).values("server_name", "expires_at")
    }

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
            connect_url = f"/api/mcp/oauth/{name}/start/" if status != MCP_STATUS_ACTIVE else None
        else:
            status = MCP_STATUS_ACTIVE
            connect_url = None

        result.append(
            {
                "server_name": name,
                "label": server.label or name.title(),
                "type": server.type,
                "status": status,
                "tool_names": server.tools or [],
                "connect_url": connect_url,
            }
        )

    return result
