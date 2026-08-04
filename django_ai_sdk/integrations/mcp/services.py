"""OAuth 2.1 + PKCE mechanics for MCP servers: discovery, dynamic client registration,
the authorization URL, the code exchange, token storage and refresh.

Connection management (listing integrations, disconnect, reconnect) is kind-agnostic
and lives in the host project's integrations endpoints, built over
IntegrationService (integrations/services.py), which dispatches to the Integration
contract. This module is only the MCP-specific OAuth plumbing those endpoints end up
calling.

Plain module-level async functions (no class namespace). Synchronous aliases for the
few functions used from sync contexts are defined at the bottom of the module.
"""

from __future__ import annotations

import logging
import secrets
from typing import TYPE_CHECKING
from urllib.parse import urlencode

import httpx
from asgiref.sync import async_to_sync
from authlib.integrations.base_client.errors import OAuthError
from django.conf import settings
from mcp.client.auth import OAuthRegistrationError
from mcp.client.auth.oauth2 import PKCEParameters
from mcp.client.auth.utils import handle_registration_response
from mcp.shared.auth import OAuthClientMetadata
from pydantic import AnyUrl

from django_ai_sdk.integrations.mcp.discovery import OAuthDiscovery, discover
from django_ai_sdk.integrations.mcp.models import MCPOAuthClient, MCPOAuthToken

if TYPE_CHECKING:
    from django_ai_sdk.integrations.mcp.schemas import MCPIntegrationConfig
    from django_ai_sdk.types import UserType

logger = logging.getLogger(__name__)


async def _mcp_config(server_name: str) -> MCPIntegrationConfig | None:
    """The MCP config for one registered integration, or None if it isn't one.

    The OAuth endpoints below are reached by server *name* (from a URL), so they need
    to resolve that back to a config. Going through the registry means a name that
    belongs to a non-MCP integration simply reads as "not an OAuth server" rather than
    exploding on a missing attribute.
    """
    from django_ai_sdk.integrations.mcp.loader import DynamicMCPIntegration
    from django_ai_sdk.integrations.registry import get_integrations

    integration = (await get_integrations([server_name])).get(server_name)
    return integration.config if isinstance(integration, DynamicMCPIntegration) else None


# ============================================================================
# OAuth PKCE helpers (pure, no I/O)
# ============================================================================


def build_pkce_params() -> tuple[str, str, str]:
    """Generate PKCE parameters: (verifier, challenge, state).

    Verifier and challenge come from the mcp SDK's PKCEParameters; state is
    generated here since the SDK's PKCE model doesn't cover it.
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
    server = await _mcp_config(server_name)
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
        if oauth_client.redirect_uri == redirect_uri:
            return oauth_client.client_id, oauth_client.get_client_secret()
        # The redirect_uri the IdP has on file for this client no longer matches
        # what we'd send it (host/scheme/path change — e.g. a URLconf move or a
        # new deployment domain). Re-registering is the only way to recover; the
        # stale client_id would otherwise get "invalid redirect_uri" forever.
        logger.info(
            "Redirect URI changed for %r (%r -> %r) — re-registering OAuth client",
            server_name,
            oauth_client.redirect_uri,
            redirect_uri,
        )

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
    oauth_client.redirect_uri = redirect_uri
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
        ValueError: token endpoint returned an error or unparseable/incomplete response
    """
    from django_ai_sdk.integrations.mcp.loader import build_oauth_client

    client = build_oauth_client(client_id, client_secret)
    try:
        async with client:
            token = await client.fetch_token(
                token_endpoint,
                grant_type="authorization_code",
                code=code,
                redirect_uri=redirect_uri,
                code_verifier=verifier,
            )
    except (OAuthError, ValueError) as e:
        raise ValueError(str(e)) from e
    if not token.get("access_token"):
        raise ValueError("Token endpoint response missing access_token")
    return dict(token)


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

    A thin wrapper over the single refresh path in loader.refresh_oauth_token: it
    just resolves the stored token and server config for this (user, server).

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

    server = await _mcp_config(server_name)
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
    server = await _mcp_config(server_name)
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

refresh_access_token_sync = async_to_sync(refresh_access_token)
