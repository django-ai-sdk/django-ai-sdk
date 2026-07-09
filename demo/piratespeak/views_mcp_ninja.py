"""Demo MCP views using Ninja Router."""

from __future__ import annotations

from typing import Literal

import httpx
from django.conf import settings
from django.contrib.auth.models import AnonymousUser
from django.http import HttpRequest
from django_ai_sdk.integrations.mcp import services as mcp_service
from django_ai_sdk.integrations.mcp.models import MCPOAuthToken
from django_ai_sdk.integrations.mcp.schemas import (
    OAuthMCPIntegrationConfig,
    StaticMCPIntegrationConfig,
    TokenMCPIntegrationConfig,
)
from ninja import Router, Schema

router = Router()


class MCPConnectionOut(Schema):
    """MCP server connection status."""

    server_name: str
    label: str
    type: str
    connected: bool | None = None
    has_token: bool = False


class MCPTestOut(Schema):
    """Result of testing an MCP connection."""

    status: Literal["connected", "refreshed", "expired", "not_connected", "error"]
    message: str


class ErrorResponse(Schema):
    """Error response."""

    detail: str


@router.get("/connections/", response=list[MCPConnectionOut], operation_id="list_connections")
async def list_connections(request: HttpRequest) -> list[MCPConnectionOut]:
    """List all MCP servers with connection status."""

    user = request.user
    if isinstance(user, AnonymousUser) or not user.is_authenticated:
        return []

    connections = await mcp_service.list_connections(user=user)
    return [
        MCPConnectionOut(
            server_name=conn.server_name,
            label=conn.label,
            type=conn.type,
            connected=conn.connected,
            has_token=conn.has_token,
        )
        for conn in connections
    ]


@router.delete(
    "/connections/{server_name}/",
    response={200: dict[str, str], 401: dict[str, str], 404: dict[str, str]},
    operation_id="disconnect",
)
async def disconnect(request: HttpRequest, server_name: str) -> tuple[int, dict[str, str]]:
    """Disconnect (revoke) an OAuth MCP connection."""
    if not request.user.is_authenticated:
        return 401, {"detail": "Not authenticated"}

    deleted = await mcp_service.disconnect(server_name, user=request.user)
    if not deleted:
        return 404, {"detail": "Not connected"}
    return 200, {"disconnected": server_name}


@router.post(
    "/connections/{server_name}/test/", response={200: MCPTestOut}, operation_id="test_connection"
)
async def test_connection(request: HttpRequest, server_name: str) -> MCPTestOut:
    """Test connectivity to an MCP server."""
    if not request.user.is_authenticated:
        return MCPTestOut(status="not_connected", message="Not authenticated")

    all_servers = getattr(settings, "AI_SDK_INTEGRATIONS", {})
    server = all_servers.get(server_name)
    # AI_SDK_INTEGRATIONS also holds non-MCP entries (API integrations, dotted paths);
    # this endpoint only tests MCP servers.
    if not isinstance(
        server,
        (StaticMCPIntegrationConfig, TokenMCPIntegrationConfig, OAuthMCPIntegrationConfig),
    ):
        return MCPTestOut(status="error", message="Server not configured")

    token: str | None = None
    refreshed = False

    if server.type == "token":
        token = server.token.get_secret_value()
    elif server.type == "oauth":
        try:
            token_obj = await MCPOAuthToken.objects.aget(user=request.user, server_name=server_name)
        except MCPOAuthToken.DoesNotExist:
            return MCPTestOut(status="not_connected", message="Not connected")

        if token_obj.is_expired():
            if not token_obj.get_refresh_token():
                return MCPTestOut(status="expired", message="Token expired, no refresh available")
            try:
                token_obj = await mcp_service.refresh_access_token(server_name, user=request.user)
                refreshed = True
            except Exception:
                return MCPTestOut(status="expired", message="Token refresh failed")

        token = token_obj.get_access_token()

    # Probe the server
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(server.url, headers=headers)

        if response.status_code == 401:
            return MCPTestOut(status="error", message="Token rejected (401)")

        status = "refreshed" if refreshed else "connected"
        message = "Token refreshed, verified" if refreshed else "Connection verified"
        return MCPTestOut(status=status, message=message)

    except httpx.ConnectError:
        return MCPTestOut(status="error", message="Server unreachable")
    except httpx.TimeoutException:
        return MCPTestOut(status="error", message="Connection timeout")
    except Exception:
        return MCPTestOut(status="error", message="Unexpected error")
