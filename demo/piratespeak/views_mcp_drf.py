"""Demo MCP views using Django REST Framework."""

from django.http import HttpRequest
from django_ai_sdk.mcp.services import (
    disconnect_sync,
    list_connections_sync,
    refresh_access_token_sync,
)
from rest_framework import routers, serializers, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response


class MCPConnectionSerializer(serializers.Serializer):
    """MCP server connection status."""

    server_name = serializers.CharField()
    label = serializers.CharField()
    type = serializers.CharField()
    connected = serializers.BooleanField(allow_null=True)
    has_token = serializers.BooleanField(default=False)


class MCPTestSerializer(serializers.Serializer):
    """Result of testing an MCP connection."""

    status = serializers.ChoiceField(
        choices=["connected", "refreshed", "expired", "not_connected", "error"]
    )
    message = serializers.CharField()


class MCPViewSet(viewsets.ViewSet):
    """MCP connection management endpoints."""

    @action(detail=False, methods=["get"])
    def connections(self, request: HttpRequest) -> Response:
        """List all MCP servers with connection status."""
        if not request.user.is_authenticated:
            return Response([])

        connections = list_connections_sync(user=request.user)
        data = [
            {
                "server_name": conn.server_name,
                "label": conn.label,
                "type": conn.type,
                "connected": conn.connected,
                "has_token": conn.has_token,
            }
            for conn in connections
        ]
        serializer = MCPConnectionSerializer(data, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=["delete"])
    def disconnect(self, request: HttpRequest, pk: str | None = None) -> Response:
        """Disconnect (revoke) an OAuth MCP connection."""
        if not request.user.is_authenticated:
            return Response({"detail": "Not authenticated"}, status=status.HTTP_401_UNAUTHORIZED)

        server_name = pk
        deleted = disconnect_sync(server_name, user=request.user)
        if not deleted:
            return Response({"detail": "Not connected"}, status=status.HTTP_404_NOT_FOUND)
        return Response({"disconnected": server_name})

    @action(detail=True, methods=["post"])
    def test(self, request: HttpRequest, pk: str | None = None) -> Response:
        """Test connectivity to an MCP server."""
        if not request.user.is_authenticated:
            result = {
                "status": "not_connected",
                "message": "Not authenticated",
            }
            return Response(MCPTestSerializer(result).data)

        import httpx
        from django.conf import settings
        from django_ai_sdk.mcp.models import MCPOAuthToken

        server_name = pk
        all_servers = getattr(settings, "AI_SDK_MCP_SERVERS", {})
        server = all_servers.get(server_name)
        if not server:
            result = {"status": "error", "message": "Server not configured"}
            return Response(MCPTestSerializer(result).data)

        token = None
        refreshed = False

        if server.type == "token":
            token = server.token
        elif server.type == "oauth":
            try:
                token_obj = MCPOAuthToken.objects.get(user=request.user, server_name=server_name)
            except MCPOAuthToken.DoesNotExist:
                result = {"status": "not_connected", "message": "Not connected"}
                return Response(MCPTestSerializer(result).data)

            if token_obj.is_expired():
                if not token_obj.get_refresh_token():
                    result = {
                        "status": "expired",
                        "message": "Token expired, no refresh available",
                    }
                    return Response(MCPTestSerializer(result).data)
                try:
                    token_obj = refresh_access_token_sync(server_name, user=request.user)
                    refreshed = True
                except Exception:
                    result = {
                        "status": "expired",
                        "message": "Token refresh failed",
                    }
                    return Response(MCPTestSerializer(result).data)

            token = token_obj.get_access_token()

        # Probe the server
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        try:
            response = httpx.get(server.url, headers=headers, timeout=5)

            if response.status_code == 401:
                result = {"status": "error", "message": "Token rejected (401)"}
                return Response(MCPTestSerializer(result).data)

            status_val = "refreshed" if refreshed else "connected"
            message = "Token refreshed, verified" if refreshed else "Connection verified"
            result = {"status": status_val, "message": message}
            return Response(MCPTestSerializer(result).data)

        except httpx.ConnectError:
            result = {"status": "error", "message": "Server unreachable"}
            return Response(MCPTestSerializer(result).data)
        except httpx.TimeoutException:
            result = {"status": "error", "message": "Connection timeout"}
            return Response(MCPTestSerializer(result).data)
        except Exception:
            result = {"status": "error", "message": "Unexpected error"}
            return Response(MCPTestSerializer(result).data)


router = routers.SimpleRouter()
router.register(r"mcp", MCPViewSet, basename="mcp")

urlpatterns = router.urls
