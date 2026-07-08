"""Service functions for MCP connection management and OAuth flows.

Plain module-level async functions (no class namespace). Synchronous aliases for the
few functions used from sync contexts (e.g. class-based views) are defined at the
bottom of the module.
"""

from __future__ import annotations

import logging
import secrets
from typing import TYPE_CHECKING
from urllib.parse import urlencode

import httpx
from asgiref.sync import async_to_sync
from django.conf import settings
from django.db import DatabaseError
from mcp.client.auth import OAuthRegistrationError, OAuthTokenError
from mcp.client.auth.oauth2 import PKCEParameters
from mcp.client.auth.utils import handle_registration_response
from mcp.shared.auth import OAuthClientMetadata
from pydantic import AnyUrl

from django_ai_sdk.integrations.base import IntegrationStatus
from django_ai_sdk.integrations.mcp.discovery import OAuthDiscovery, discover
from django_ai_sdk.integrations.mcp.models import MCPOAuthClient, MCPOAuthToken
from django_ai_sdk.integrations.mcp.schemas import (
    ConnectionOut,
    OAuthMCPIntegrationConfig,
    StaticMCPIntegrationConfig,
    TokenMCPIntegrationConfig,
)
from django_ai_sdk.integrations.registry import get_integrations

if TYPE_CHECKING:
    from django_ai_sdk.types import UserType

logger = logging.getLogger(__name__)


def _get_mcp_servers() -> dict:
    """Get the MCP-backed entries from AI_SDK_INTEGRATIONS.

    AI_SDK_INTEGRATIONS may also contain hand-written API-backed integrations
    (registered as a dotted class path rather than an MCP config object) — those
    have no OAuth/connection concept, so this MCP-specific service layer filters
    them out rather than assuming every entry has a `.type`/`.label`.
    """
    configured: dict = getattr(settings, "AI_SDK_INTEGRATIONS", {})
    mcp_types = (StaticMCPIntegrationConfig, TokenMCPIntegrationConfig, OAuthMCPIntegrationConfig)
    return {name: value for name, value in configured.items() if isinstance(value, mcp_types)}


# ============================================================================
# Connection Management
# ============================================================================


async def list_connections(*, user: UserType) -> list[ConnectionOut]:
    """List all MCP servers with connection status for the user."""
    all_servers = _get_mcp_servers()
    if not all_servers:
        return []

    # Bulk-fetch all OAuth tokens for this user in a single query.
    oauth_tokens: dict[str, MCPOAuthToken] = {}
    try:
        async for t in MCPOAuthToken.objects.filter(user=user):
            oauth_tokens[t.server_name] = t
    except DatabaseError:
        logger.warning(
            "Failed to load MCP OAuth token connections for user=%s; continuing with no connected servers.",
            getattr(user, "pk", None),
            exc_info=True,
        )

    integrations = get_integrations(list(all_servers))

    result = []
    for server_name, server_config in all_servers.items():
        if server_config.type == "oauth":
            token_obj = oauth_tokens.get(server_name)
            is_connected: bool | None = not token_obj.is_expired() if token_obj else False
        else:
            is_connected = None

        integration = integrations.get(server_name)
        status = (
            await integration.get_status(user=user)
            if integration is not None
            else IntegrationStatus.DISCONNECTED
        )

        result.append(
            ConnectionOut(
                server_name=server_name,
                label=server_config.label or server_name.title(),
                type=server_config.type,
                connected=is_connected,
                has_token=server_name in oauth_tokens,
                status=status,
            )
        )

    return result


async def disconnect(server_name: str, *, user: UserType) -> bool:
    """Revoke OAuth token for a server. Returns True if deleted, False if not found."""
    if not user:
        return False

    deleted, _ = await MCPOAuthToken.objects.filter(user=user, server_name=server_name).adelete()
    return deleted > 0


async def reconnect(server_name: str, *, user: UserType) -> IntegrationStatus | None:
    """Reset a degraded integration's retry state and immediately make a real attempt,
    so the caller gets the actual outcome rather than a blind promise — resetting the
    state doesn't mean the underlying problem (e.g. a still-wrong URL) is fixed. Returns
    None if the server isn't configured."""
    integration = get_integrations([server_name]).get(server_name)
    if integration is None:
        return None
    await integration.reconnect(user=user)
    return await integration.get_status(user=user)


# ============================================================================
# OAuth PKCE helpers (pure, no I/O)
# ============================================================================


def build_pkce_params() -> tuple[str, str, str]:
    """Generate PKCE parameters: (verifier, challenge, state).

    Verifier/challenge come from the mcp SDK's ``PKCEParameters``; ``state`` is
    generated here (the SDK's PKCE model doesn't cover it).
    """
    pkce = PKCEParameters.generate()
    return pkce.code_verifier, pkce.code_challenge, secrets.token_urlsafe(32)


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
# OAuth I/O helpers
# ============================================================================


async def get_or_register_client(
    server_name: str, redirect_uri: str, discovery: OAuthDiscovery
) -> tuple[str, str]:
    """Get or register the OAuth client, performing dynamic registration if needed.

    Returns (client_id, client_secret).
    """
    server = _get_mcp_servers().get(server_name)
    if not server or server.type != "oauth":
        raise ValueError(f"Server {server_name!r} not found or not OAuth type")

    # Static credentials win when configured.
    if server.client_id:
        return server.client_id, server.client_secret.get_secret_value()

    # Otherwise use dynamic registration (RFC 7591).
    oauth_client, created = await MCPOAuthClient.objects.aget_or_create(
        server_name=server_name,
        defaults={"redirect_uri": redirect_uri},
    )
    if not created and oauth_client.client_id:
        return oauth_client.client_id, oauth_client.get_client_secret()

    if not discovery.registration_endpoint:
        raise ValueError(
            f"Server {server_name!r} has no registration_endpoint; "
            "provide static client_id/client_secret instead."
        )

    client_metadata = OAuthClientMetadata(
        client_name=getattr(settings, "AI_SDK_MCP_CLIENT_NAME", "MCP OAuth Client"),
        redirect_uris=[AnyUrl(redirect_uri)],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
    )
    async with httpx.AsyncClient(timeout=10) as http_client:
        reg_response = await http_client.post(
            discovery.registration_endpoint,
            json=client_metadata.model_dump(by_alias=True, mode="json", exclude_none=True),
            headers={"Content-Type": "application/json"},
        )
    try:
        client_info = await handle_registration_response(reg_response)
    except OAuthRegistrationError as e:
        raise ValueError(str(e)) from e

    client_id = client_info.client_id
    client_secret = client_info.client_secret or ""
    if not client_id:
        raise ValueError("No client_id in registration response")

    logger.info("Dynamically registered client for %r: client_id=%s", server_name, client_id)
    oauth_client.set_credentials(client_id, client_secret)
    await oauth_client.asave()
    return client_id, client_secret


async def exchange_token(
    token_endpoint: str,
    code: str,
    redirect_uri: str,
    verifier: str,
    client_id: str,
    client_secret: str = "",
) -> dict:
    """Exchange an authorization code for an access token.

    Raises:
        httpx.HTTPError: token endpoint transport/status error
        ValueError: token endpoint returned an unparseable response
    """
    from django_ai_sdk.integrations.mcp.loader import post_token_request

    try:
        token = await post_token_request(
            token_endpoint,
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "code_verifier": verifier,
                "client_id": client_id,
            },
            client_id,
            client_secret,
        )
    except OAuthTokenError as e:
        raise ValueError(str(e)) from e
    return token.model_dump()


async def store_token(user: UserType, server_name: str, token_response: dict) -> MCPOAuthToken:
    """Store an OAuth token for a user."""
    if not user:
        raise ValueError("User required to store token")

    token_obj, _ = await MCPOAuthToken.objects.aget_or_create(user=user, server_name=server_name)
    token_obj.set_tokens(token_response)
    await token_obj.asave()
    logger.info("Stored token for %r user=%s", server_name, user)
    return token_obj


async def refresh_access_token(server_name: str, *, user: UserType) -> MCPOAuthToken:
    """Refresh a user's stored OAuth access token.

    A thin wrapper over the single refresh path in ``loader.refresh_oauth_token`` —
    it just resolves the stored token and server config for this (user, server).

    Raises:
        ValueError: no token, server not configured/OAuth, or the refresh failed.
    """
    if not user:
        raise ValueError("User required")

    from django_ai_sdk.integrations.mcp.loader import refresh_oauth_token

    try:
        token_obj = await MCPOAuthToken.objects.aget(user=user, server_name=server_name)
    except MCPOAuthToken.DoesNotExist:
        raise ValueError(f"No token for server {server_name!r}") from None

    server = _get_mcp_servers().get(server_name)
    if not server or server.type != "oauth":
        raise ValueError(f"Server {server_name!r} not found or not OAuth type")

    refreshed = await refresh_oauth_token(token_obj, server)
    if refreshed is None:
        raise ValueError(f"Token refresh failed for {server_name!r}")
    logger.info("Refreshed OAuth token for %r user=%s", server_name, user)
    return refreshed


# ============================================================================
# OAuth discovery
# ============================================================================


async def get_oauth_discovery(server_name: str) -> OAuthDiscovery:
    """Get OAuth discovery for a server (static endpoints if configured, else RFC 9728)."""
    server = _get_mcp_servers().get(server_name)
    if not server or server.type != "oauth":
        raise ValueError(f"Server {server_name!r} not found or not OAuth type")

    if server.authorization_endpoint and server.token_endpoint:
        return OAuthDiscovery(
            authorization_endpoint=server.authorization_endpoint,
            token_endpoint=server.token_endpoint,
        )

    return await discover(server.oauth_discovery_url or server.url)


# ============================================================================
# Sync aliases for use in synchronous contexts
# ============================================================================

list_connections_sync = async_to_sync(list_connections)
disconnect_sync = async_to_sync(disconnect)
refresh_access_token_sync = async_to_sync(refresh_access_token)
