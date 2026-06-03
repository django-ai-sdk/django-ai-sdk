"""Unified MCP tool loader — static, token, and OAuth server types."""

from __future__ import annotations

import importlib
import logging
import time
from typing import TYPE_CHECKING, Any

import httpx
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.mcp.schemas import OAuthMCPServer, StaticMCPServer, TokenMCPServer

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.mcp.models import MCPOAuthToken

logger = logging.getLogger(__name__)

# TTL cache for static and token servers — keyed by server name.
# OAuth is per-user so is not cached here.
_tool_cache: dict[str, tuple[float, list]] = {}


def _ttl() -> int:
    return getattr(settings, "AI_SDK_MCP_CACHE_TTL", 300)


async def load_mcp_tools(
    config: dict[str, Any],
    user: AbstractBaseUser | AnonymousUser | None = None,
) -> list[Any]:
    """
    Load tool objects from all MCP servers in config.

    Returns generic tool objects from the configured AI_SDK_MCP_BACKEND.

    Static and token servers are cached in-process with TTL.
    OAuth servers connect per-request using the user's stored token.
    """
    tools: list[Any] = []
    for name, server in config.items():
        try:
            server_tools = await _load_server(name, server, user)
            tools.extend(server_tools)
        except Exception:
            logger.exception("Failed to load MCP tools for server %r", name)
    return tools


async def _load_server(
    name: str, server: Any, user: AbstractBaseUser | AnonymousUser | None
) -> list[Any]:
    if isinstance(server, StaticMCPServer):
        return await _load_cached(name, server.url, token=None, tools=server.tools or None)
    if isinstance(server, TokenMCPServer):
        return await _load_cached(name, server.url, token=server.token, tools=server.tools or None)
    if isinstance(server, OAuthMCPServer):
        return await _load_oauth(name, server, user)
    logger.warning("Unrecognised MCP server type for %r: %s", name, type(server).__name__)
    return []


async def _load_cached(
    name: str,
    url: str,
    token: str | None,
    tools: list[str] | None,
) -> list[Any]:
    ttl = _ttl()
    now = time.monotonic()

    if ttl > 0 and name in _tool_cache:
        expires_at, cached = _tool_cache[name]
        if now < expires_at:
            logger.debug("MCP cache hit for %r (%d tools)", name, len(cached))
            return cached

    logger.info("Connecting to MCP server %r", name)
    result = await _connect(url, token, tools)

    if ttl > 0:
        _tool_cache[name] = (now + ttl, result)

    return result


async def _load_oauth(
    name: str, server: OAuthMCPServer, user: AbstractBaseUser | AnonymousUser | None
) -> list[Any]:
    if not user:
        logger.debug("No user provided for OAuth server %r", name)
        return []

    from django_ai_sdk.mcp.models import MCPOAuthToken

    try:
        token_obj = await MCPOAuthToken.objects.aget(user_id=user.pk, server_name=name)
    except MCPOAuthToken.DoesNotExist:
        logger.debug("No OAuth token stored for user %s / server %r", user.pk, name)
        return []
    except Exception as e:
        logger.warning("Error loading OAuth token for user %s / server %r: %s", user.pk, name, e)
        return []

    if token_obj.is_expired():
        logger.info("Access token expired for %r — attempting refresh", name)
        token_obj = await refresh_oauth_token(token_obj, server)
        if token_obj is None:
            return []

    access_token = token_obj.get_access_token()
    if not access_token:
        logger.warning("Empty access token for user %s / server %r", user.pk, name)
        return []

    logger.info("Connecting to OAuth MCP server %r for user %s", name, user.pk)
    return await _connect(server.url, access_token, server.tools or None)


async def refresh_oauth_token(
    token_obj: MCPOAuthToken, server: OAuthMCPServer
) -> MCPOAuthToken | None:
    """Attempt a refresh_token grant. Returns the updated token_obj or None on failure."""
    refresh_token = token_obj.get_refresh_token()
    if not refresh_token:
        logger.warning("No refresh token available for server %r", token_obj.server_name)
        return None

    from django_ai_sdk.mcp.discovery import discover
    from django_ai_sdk.mcp.models import MCPOAuthClient

    try:
        discovery = await discover(server.url)
    except Exception as e:
        logger.error(
            "Cannot discover token endpoint for refresh of %r: %s", token_obj.server_name, e
        )
        return None

    # Get credentials: dynamically-registered client from DB, or static from schema
    client_id = server.client_id
    client_secret = server.client_secret
    try:
        oauth_client = await MCPOAuthClient.objects.aget(server_name=token_obj.server_name)
        client_id = oauth_client.client_id
        client_secret = oauth_client.get_client_secret()
    except MCPOAuthClient.DoesNotExist:
        logger.debug(
            "No dynamically registered OAuth client for %r, using static credentials",
            token_obj.server_name,
        )

    payload: dict[str, str] = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            if client_secret:
                payload.pop("client_id")
                response = await client.post(
                    discovery.token_endpoint,
                    data=payload,
                    auth=(client_id, client_secret),
                )
            else:
                response = await client.post(discovery.token_endpoint, data=payload)
        response.raise_for_status()
    except httpx.HTTPError as e:
        logger.error("Token refresh failed for server %r: %s", token_obj.server_name, e)
        return None

    try:
        token_data = response.json()
    except ValueError as e:
        logger.error("Invalid JSON in refresh response for %r: %s", token_obj.server_name, e)
        return None

    if "access_token" not in token_data:
        logger.error("No access_token in refresh response for %r", token_obj.server_name)
        return None

    token_obj.set_tokens(token_data)
    await token_obj.asave()
    logger.info("Refreshed OAuth token for server %r", token_obj.server_name)
    return token_obj


def _backend_path() -> str:
    path = getattr(settings, "AI_SDK_MCP_BACKEND", None)
    if not path:
        raise ImproperlyConfigured(
            "AI_SDK_MCP_BACKEND is not set. "
            "Add it to your settings, e.g.:\n"
            "  AI_SDK_MCP_BACKEND = 'django_ai_sdk.mcp.backends.haystack'\n"
            "Or implement a custom backend by implementing the MCPBackend protocol "
            "from django_ai_sdk.mcp.backends."
        )
    return path


async def _connect(url: str, token: str | None, tools: list[str] | None) -> list[Any]:
    """Connect to an MCP server via the configured AI_SDK_MCP_BACKEND."""
    backend = importlib.import_module(_backend_path())
    result = await backend.connect(url, token, tools)
    logger.info("Loaded %d tool(s) from %s", len(result), url)
    return result
