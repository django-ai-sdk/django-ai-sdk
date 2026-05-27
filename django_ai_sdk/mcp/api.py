"""Plain Django views for MCP connection management."""

from __future__ import annotations

from django.conf import settings
from django.http import HttpRequest, JsonResponse
from django.views.decorators.http import require_http_methods

from django_ai_sdk.mcp.models import MCPOAuthToken


@require_http_methods(["GET"])
async def list_connections(request: HttpRequest) -> JsonResponse:
    """GET connections/ — list all MCP servers with type and connection status."""
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Not authenticated"}, status=401)

    all_servers = getattr(settings, "AI_SDK_MCP_SERVERS", {})

    connected = set(
        [
            sn
            async for sn in MCPOAuthToken.objects.filter(user=request.user).values_list(
                "server_name", flat=True
            )
        ]
    )

    # Derive the SDK mount base from the request's own path (/…/connections/ → /…/).
    # This is always correct regardless of where the SDK URLs are mounted.
    base = request.path.removesuffix("connections/")

    result = []
    for name, server in all_servers.items():
        entry: dict = {
            "server_name": name,
            "label": server.label or name.title(),
            "type": server.type,
        }
        if server.type == "oauth":
            entry["connected"] = name in connected
            entry["connect_url"] = f"{base}oauth/{name}/start/"
            entry["disconnect_url"] = f"{base}connections/{name}/"
        result.append(entry)

    return JsonResponse(result, safe=False)


@require_http_methods(["DELETE"])
async def disconnect_server(request: HttpRequest, server_name: str) -> JsonResponse:
    """DELETE connections/<server_name>/ — revoke stored OAuth token."""
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Not authenticated"}, status=401)

    deleted, _ = await MCPOAuthToken.objects.filter(
        user=request.user, server_name=server_name
    ).adelete()
    if deleted:
        return JsonResponse({"disconnected": server_name})
    return JsonResponse({"error": "Not connected"}, status=404)
