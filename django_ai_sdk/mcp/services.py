"""Service layer for MCP connection management and OAuth flows."""

from __future__ import annotations

import base64
import hashlib
import logging
import secrets
from typing import TYPE_CHECKING
from urllib.parse import urlencode

import httpx
from asgiref.sync import async_to_sync
from django.conf import settings

from django_ai_sdk.mcp.discovery import OAuthDiscovery, discover
from django_ai_sdk.mcp.models import MCPOAuthClient, MCPOAuthToken
from django_ai_sdk.mcp.schemas import ConnectionOut

if TYPE_CHECKING:
    from django.contrib.auth.models import AbstractUser

logger = logging.getLogger(__name__)


def _b64url(data: bytes) -> str:
    """Base64url encode without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _get_mcp_servers() -> dict:
    """Get MCP servers from settings."""
    return getattr(settings, "AI_SDK_MCP_SERVERS", {})


class MCPService:
    """Service for MCP connection management and OAuth flows.

    All methods are async. Sync aliases are provided at module level for use in
    synchronous contexts (e.g., class-based views).
    """

    # ============================================================================
    # Connection Management
    # ============================================================================

    @staticmethod
    async def list_connections(*, user: AbstractUser | None) -> list[ConnectionOut]:
        """List all MCP servers with connection status for the user."""
        all_servers = _get_mcp_servers()
        if not all_servers:
            return []

        # Get connected OAuth servers for this user
        connected = set()
        try:
            async for sn in MCPOAuthToken.objects.filter(user=user).values_list(
                "server_name", flat=True
            ):
                connected.add(sn)
        except Exception:
            pass

        result = []
        for server_name, server_config in all_servers.items():
            # Determine if this server is connected
            is_connected = None
            if server_config.type == "oauth":
                if server_name in connected:
                    # Check if token is expired
                    try:
                        token_obj = await MCPOAuthToken.objects.aget(
                            user=user, server_name=server_name
                        )
                        is_connected = not token_obj.is_expired()
                    except MCPOAuthToken.DoesNotExist:
                        is_connected = False
                else:
                    is_connected = False
            else:
                # Static/token servers don't have OAuth tokens, so no "connection" state
                is_connected = None

            result.append(
                ConnectionOut(
                    server_name=server_name,
                    label=server_config.label or server_name.title(),
                    type=server_config.type,
                    connected=is_connected,
                    has_token=server_name in connected,
                )
            )

        return result

    @staticmethod
    async def disconnect(server_name: str, *, user: AbstractUser | None) -> bool:
        """Revoke OAuth token for a server. Returns True if deleted, False if not found."""
        if not user:
            return False

        deleted, _ = await MCPOAuthToken.objects.filter(
            user=user, server_name=server_name
        ).adelete()
        return deleted > 0

    # ============================================================================
    # OAuth PKCE Helpers (pure, no I/O)
    # ============================================================================

    @staticmethod
    def build_pkce_params() -> tuple[str, str, str]:
        """Generate PKCE parameters: (verifier, challenge, state)."""
        verifier = secrets.token_urlsafe(96)
        challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
        state = secrets.token_urlsafe(32)
        return verifier, challenge, state

    @staticmethod
    def build_auth_url(
        discovery: OAuthDiscovery,
        client_id: str,
        redirect_uri: str,
        state: str,
        challenge: str,
        scope: str = "",
    ) -> str:
        """Build the OAuth authorization URL."""
        auth_params = {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        if scope:
            auth_params["scope"] = scope

        return discovery.authorization_endpoint + "?" + urlencode(auth_params)

    # ============================================================================
    # OAuth I/O Helpers
    # ============================================================================

    @staticmethod
    async def get_or_register_client(
        server_name: str, redirect_uri: str, discovery: OAuthDiscovery
    ) -> tuple[str, str]:
        """Get or register OAuth client, perform dynamic registration if needed.

        Returns (client_id, client_secret).
        """
        all_servers = _get_mcp_servers()
        server = all_servers.get(server_name)
        if not server or server.type != "oauth":
            raise ValueError(f"Server {server_name!r} not found or not OAuth type")

        # Use static credentials if available
        if server.client_id:
            return server.client_id, server.client_secret or ""

        # Otherwise use dynamic registration
        oauth_client, created = await MCPOAuthClient.objects.aget_or_create(
            server_name=server_name,
            defaults={"redirect_uri": redirect_uri},
        )

        if not created and oauth_client.client_id:
            return oauth_client.client_id, oauth_client.get_client_secret()

        # Perform dynamic registration (RFC 7591)
        if not discovery.registration_endpoint:
            raise ValueError(
                f"Server {server_name!r} has no registration_endpoint; "
                "provide static client_id/client_secret instead."
            )

        client_name = getattr(settings, "AI_SDK_MCP_CLIENT_NAME", "MCP OAuth Client")
        registration_data = {
            "client_name": client_name,
            "redirect_uris": [redirect_uri],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
        }
        async with httpx.AsyncClient(timeout=10) as http_client:
            reg_response = await http_client.post(
                discovery.registration_endpoint,
                json=registration_data,
            )
        reg_response.raise_for_status()
        reg_data = reg_response.json()
        client_id = reg_data.get("client_id")
        client_secret = reg_data.get("client_secret", "")
        if not client_id:
            raise ValueError("No client_id in registration response")

        logger.info("Dynamically registered client for %r: client_id=%s", server_name, client_id)

        oauth_client.set_credentials(client_id, client_secret)
        await oauth_client.asave()

        return client_id, client_secret

    @staticmethod
    async def exchange_token(
        token_endpoint: str,
        code: str,
        redirect_uri: str,
        verifier: str,
        client_id: str,
        client_secret: str = "",
    ) -> dict:
        """Exchange authorization code for access token."""
        token_data: dict[str, str] = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": verifier,
            "client_id": client_id,
        }

        async with httpx.AsyncClient(timeout=10) as client:
            if client_secret:
                token_data.pop("client_id")
                response = await client.post(
                    token_endpoint,
                    data=token_data,
                    auth=(client_id, client_secret),
                )
            else:
                response = await client.post(token_endpoint, data=token_data)

        response.raise_for_status()
        token_response = response.json()

        if "access_token" not in token_response:
            raise ValueError("No access_token in token response")

        return token_response

    @staticmethod
    async def store_token(
        user: AbstractUser | None, server_name: str, token_response: dict
    ) -> MCPOAuthToken:
        """Store OAuth token for user."""
        if not user:
            raise ValueError("User required to store token")

        token_obj, _ = await MCPOAuthToken.objects.aget_or_create(
            user=user,
            server_name=server_name,
        )
        token_obj.set_tokens(token_response)
        await token_obj.asave()
        logger.info("Stored token for %r user=%s", server_name, user)
        return token_obj

    @staticmethod
    async def refresh_access_token(server_name: str, *, user: AbstractUser | None) -> MCPOAuthToken:
        """Refresh the OAuth access token using the stored refresh_token.

        Raises:
            ValueError: no token, no refresh_token, or bad token response
            httpx.HTTPStatusError: token endpoint rejected the grant
        """
        if not user:
            raise ValueError("User required")

        try:
            token_obj = await MCPOAuthToken.objects.aget(user=user, server_name=server_name)
        except MCPOAuthToken.DoesNotExist:
            raise ValueError(f"No token for server {server_name!r}")

        refresh_token = token_obj.get_refresh_token()
        if not refresh_token:
            raise ValueError(f"No refresh_token stored for {server_name!r}")

        discovery = await MCPService.get_oauth_discovery(server_name)

        # Resolve client credentials — dynamic registration takes precedence over static config
        all_servers = _get_mcp_servers()
        server = all_servers.get(server_name)
        if not server or server.type != "oauth":
            raise ValueError(f"Server {server_name!r} not found or not OAuth type")

        client_id = getattr(server, "client_id", "") or ""
        client_secret = getattr(server, "client_secret", "") or ""
        try:
            oauth_client = await MCPOAuthClient.objects.aget(server_name=server_name)
            client_id = oauth_client.client_id
            client_secret = oauth_client.get_client_secret()
        except MCPOAuthClient.DoesNotExist:
            pass

        token_data: dict[str, str] = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
        }

        async with httpx.AsyncClient(timeout=10) as client:
            if client_secret:
                token_data.pop("client_id")
                response = await client.post(
                    discovery.token_endpoint, data=token_data, auth=(client_id, client_secret)
                )
            else:
                response = await client.post(discovery.token_endpoint, data=token_data)

        response.raise_for_status()
        token_response = response.json()
        if "access_token" not in token_response:
            raise ValueError("No access_token in refresh response")

        token_obj = await MCPService.store_token(
            user=user, server_name=server_name, token_response=token_response
        )
        logger.info("Refreshed OAuth token for %r user=%s", server_name, user)
        return token_obj

    # ============================================================================
    # OAuth Discovery
    # ============================================================================

    @staticmethod
    async def get_oauth_discovery(server_name: str) -> OAuthDiscovery:
        """Get OAuth discovery for a server."""
        all_servers = _get_mcp_servers()
        server = all_servers.get(server_name)
        if not server or server.type != "oauth":
            raise ValueError(f"Server {server_name!r} not found or not OAuth type")

        if server.authorization_endpoint and server.token_endpoint:
            return OAuthDiscovery(
                authorization_endpoint=server.authorization_endpoint,
                token_endpoint=server.token_endpoint,
            )

        discovery_url = server.oauth_discovery_url or server.url
        return await discover(discovery_url)


# ============================================================================
# Sync aliases for use in synchronous contexts
# ============================================================================

list_connections_sync = async_to_sync(MCPService.list_connections)
disconnect_sync = async_to_sync(MCPService.disconnect)
get_oauth_discovery_sync = async_to_sync(MCPService.get_oauth_discovery)
get_or_register_client_sync = async_to_sync(MCPService.get_or_register_client)
exchange_token_sync = async_to_sync(MCPService.exchange_token)
store_token_sync = async_to_sync(MCPService.store_token)
refresh_access_token_sync = async_to_sync(MCPService.refresh_access_token)
