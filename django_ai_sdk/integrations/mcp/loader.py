"""MCP-backed Integration — connects to a remote MCP server over Streamable HTTP.

Discovery (connect + list_tools) is the one place this backend touches the network,
and it's the part that used to sit directly on the chat-request critical path. Every
MCPIntegration instance owns a ResilientCache (stale-while-revalidate + circuit
breaker — see django_ai_sdk.integrations.base) so a slow or dead MCP server can add
at most one bounded, cache-miss-only delay, never a per-turn one.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import httpx
from django.conf import settings
from mcp.client.auth import OAuthTokenError
from mcp.client.auth.utils import handle_token_response_scopes

from django_ai_sdk.integrations.base import Integration, IntegrationStatus, ResilientCache
from django_ai_sdk.integrations.mcp.schemas import (
    OAuthMCPIntegrationConfig,
    StaticMCPIntegrationConfig,
    TokenMCPIntegrationConfig,
)

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser
    from mcp.shared.auth import OAuthToken

    from django_ai_sdk.assistant import Assistant
    from django_ai_sdk.integrations.mcp.models import MCPOAuthToken

logger = logging.getLogger(__name__)


class MCPIntegration(Integration):
    """One configured MCP server, exposed through the Integration contract.

    Static/token servers cache by server name only — the tool list doesn't vary per
    user. OAuth servers cache by (server name, user id), since each user has their
    own token/session; a user with no stored token (or an unrefreshable one) simply
    gets no tools from this integration, rather than triggering a connect attempt.

    Each instance owns its own ResilientCache rather than sharing one process-wide
    global — safe because `django_ai_sdk.integrations.registry.get_all_integrations()`
    already builds exactly one MCPIntegration per configured server name and caches
    it for the life of the process, so instance-scoped state has the same effective
    lifetime and sharing as a module-level singleton would, with none of the hidden
    global-mutable-state downsides (e.g. tests that construct their own instances get
    a naturally isolated cache, with no risk of bleeding into another test's).
    """

    def __init__(
        self,
        name: str,
        config: StaticMCPIntegrationConfig | TokenMCPIntegrationConfig | OAuthMCPIntegrationConfig,
    ) -> None:
        self.name = name
        self._timeout = getattr(settings, "AI_SDK_INTEGRATION_TIMEOUT", 5)
        self._cache = ResilientCache(
            ttl=getattr(settings, "AI_SDK_INTEGRATION_CACHE_TTL", 300),
            timeout=self._timeout,
            cb_cooldown=getattr(settings, "AI_SDK_INTEGRATION_CB_COOLDOWN", 60),
        )
        self.label = config.label or name.title()
        self.config = config

    @property
    def kind(self) -> str:
        return self.config.type

    async def get_tool_names(
        self, user: AbstractBaseUser | AnonymousUser | None = None
    ) -> list[str]:
        """Explicit allow-list from config when one's set — free, no cache touch.
        With auto-discovery (no allow-list configured), there's no name to report
        without asking the server what it actually offers, so this falls back to
        the same cached get_tools() every other caller uses: a cache hit whenever
        something else in the same request (e.g. get_status()) already primed it,
        and otherwise the same bounded live fetch get_tools() always pays anyway —
        never a second, additional cost beyond what discovery already costs."""
        if self.config.tools:
            return self.config.tools
        return [t.name for t in await self.get_tools(user)]

    def _cache_key(self, user: AbstractBaseUser | AnonymousUser | None) -> Any:
        if isinstance(self.config, OAuthMCPIntegrationConfig):
            user_id = getattr(user, "pk", None)
            if user_id is None:
                return None
            return (self.name, user_id)
        return self.name

    async def get_tools(
        self,
        user: AbstractBaseUser | AnonymousUser | None = None,
        assistant: Assistant | None = None,
        thread_id: str = "",
    ) -> list[Any]:
        # `assistant`/`thread_id` are unused here — MCP tools are discovered from
        # the remote server as-is, with no per-call model/context to inject.
        # Accepted only for signature consistency with the Integration ABC.
        key = self._cache_key(user)
        if key is None:
            return []
        return await self._cache.get(key, lambda: self._fetch(user))

    async def get_status(
        self,
        user: AbstractBaseUser | AnonymousUser | None = None,
        assistant: Assistant | None = None,
    ) -> IntegrationStatus:
        """Report real, attempted status — a wrong/invalid token must show as
        DEGRADED, not ACTIVE, even if nothing has connected to this server yet.

        Forces a real attempt via get_tools() when the cache doesn't already have
        one on file for this key. That call is bounded/cached like any other — this
        just guarantees "active" always means "the last real attempt succeeded",
        never "we simply never checked".
        """
        if not isinstance(self.config, OAuthMCPIntegrationConfig):
            await self.get_tools(user)
            return self._cache.status_for(self._cache_key(user))

        token = await self._get_oauth_token(user)
        if token is None:
            return IntegrationStatus.DISCONNECTED
        if token.is_expired():
            return IntegrationStatus.EXPIRED

        await self.get_tools(user)
        # A non-None token already required a valid user.pk, so _cache_key(user) here
        # can't be None.
        return self._cache.status_for(self._cache_key(user))

    async def reconnect(self, user: AbstractBaseUser | AnonymousUser | None = None) -> None:
        """Reset this integration to a fresh, never-attempted state and retry now.

        A DEGRADED integration already recovers on its own (the breaker half-opens
        after a cooldown), so this is an optional "retry immediately" — for a staff
        action, a management command, or a "Reconnect" button — that clears the cached
        value and resets the breaker rather than waiting out the cooldown.
        """
        key = self._cache_key(user)
        if key is not None:
            await self._cache.invalidate(key)

    async def warm(self) -> None:
        """Eagerly populate the cache for static/token servers at process startup."""
        if isinstance(self.config, OAuthMCPIntegrationConfig):
            return  # no per-user variance to warm without a specific user
        await self._cache.warm(self.name, lambda: self._fetch(None))

    async def _fetch(self, user: AbstractBaseUser | AnonymousUser | None) -> list[Any]:
        if isinstance(self.config, StaticMCPIntegrationConfig):
            return await _connect(
                self.config.url, token=None, tools=self.config.tools or None, timeout=self._timeout
            )
        if isinstance(self.config, TokenMCPIntegrationConfig):
            return await _connect(
                self.config.url,
                token=self.config.token.get_secret_value(),
                tools=self.config.tools or None,
                timeout=self._timeout,
            )
        return await self._fetch_oauth(user)

    async def _get_oauth_token(
        self, user: AbstractBaseUser | AnonymousUser | None
    ) -> MCPOAuthToken | None:
        if user is None:
            return None
        from django_ai_sdk.integrations.mcp.models import MCPOAuthToken

        try:
            return await MCPOAuthToken.objects.aget(user_id=user.pk, server_name=self.name)
        except MCPOAuthToken.DoesNotExist:
            return None

    async def _fetch_oauth(self, user: AbstractBaseUser | AnonymousUser | None) -> list[Any]:
        token_obj = await self._get_oauth_token(user)
        if token_obj is None:
            return []

        if token_obj.is_expired():
            logger.info("Access token expired for %r — attempting refresh", self.name)
            token_obj = await refresh_oauth_token(token_obj, self.config)
            if token_obj is None:
                return []

        access_token = token_obj.get_access_token()
        if not access_token:
            return []

        logger.info("Connecting to OAuth MCP server %r for user %s", self.name, user.pk)
        return await _connect(
            self.config.url, access_token, self.config.tools or None, timeout=self._timeout
        )


async def resolve_client_credentials(
    server_name: str, config: OAuthMCPIntegrationConfig
) -> tuple[str, str]:
    """Return (client_id, client_secret) for an OAuth server.

    A dynamically-registered client (RFC 7591, stored in MCPOAuthClient) takes
    precedence over static credentials configured on the integration. This is the
    single source of truth for credential resolution — the OAuth start/refresh
    paths all go through it.
    """
    from django_ai_sdk.integrations.mcp.models import MCPOAuthClient

    try:
        oauth_client = await MCPOAuthClient.objects.aget(server_name=server_name)
        return oauth_client.client_id, oauth_client.get_client_secret()
    except MCPOAuthClient.DoesNotExist:
        logger.debug(
            "No dynamically registered OAuth client for %r, using static credentials",
            server_name,
        )
        return config.client_id, config.client_secret.get_secret_value()


async def post_token_request(
    token_endpoint: str, data: dict[str, str], client_id: str, client_secret: str
) -> OAuthToken:
    """POST an OAuth token grant and return the parsed, validated token.

    Confidential clients authenticate with HTTP Basic (client_secret); public
    clients send client_id in the body. Raises on transport error (httpx) or an
    unparseable token response (OAuthTokenError) — callers decide how to handle.
    """
    async with httpx.AsyncClient(timeout=10) as client:
        if client_secret:
            body = {k: v for k, v in data.items() if k != "client_id"}
            response = await client.post(token_endpoint, data=body, auth=(client_id, client_secret))
        else:
            response = await client.post(token_endpoint, data=data)
    response.raise_for_status()
    return await handle_token_response_scopes(response)


async def refresh_oauth_token(
    token_obj: MCPOAuthToken, config: OAuthMCPIntegrationConfig
) -> MCPOAuthToken | None:
    """The single refresh_token grant. Returns the updated token_obj, or None on failure.

    Every refresh in the codebase (the loader's own lazy refresh, the OAuth views,
    and the refresh_mcp_tokens command) funnels through here.
    """
    refresh_token = token_obj.get_refresh_token()
    if not refresh_token:
        logger.warning("No refresh token available for server %r", token_obj.server_name)
        return None

    from django_ai_sdk.integrations.mcp.discovery import discover

    try:
        discovery = await discover(config.url)
    except Exception as e:
        logger.error(
            "Cannot discover token endpoint for refresh of %r: %s", token_obj.server_name, e
        )
        return None

    client_id, client_secret = await resolve_client_credentials(token_obj.server_name, config)
    try:
        token = await post_token_request(
            discovery.token_endpoint,
            {"grant_type": "refresh_token", "refresh_token": refresh_token, "client_id": client_id},
            client_id,
            client_secret,
        )
    except (httpx.HTTPError, OAuthTokenError) as e:
        logger.error("Token refresh failed for server %r: %s", token_obj.server_name, e)
        return None

    token_obj.set_tokens(token.model_dump())
    await token_obj.asave()
    logger.info("Refreshed OAuth token for server %r", token_obj.server_name)
    return token_obj


async def _connect(
    url: str, token: str | None, tools: list[str] | None, timeout: float
) -> list[Any]:
    """Connect to an MCP server and return Haystack tool objects.

    ``timeout`` is passed straight through as MCPToolset's own connection_timeout
    (discovery only — tool *invocation* keeps MCPToolset's own 30s default, a
    separate concern), rather than left at its 30s default: MCPToolset's connect
    happens inside a plain blocking call on a background thread (via haystack's own
    AsyncExecutor), which our caller's asyncio-level timeout (ResilientCache) has no
    power to interrupt — only MCPToolset's own internal timeout can actually abort
    and clean up the connection attempt. Keeping the two in sync avoids
    threads/connections lingering past the point where we've already given up and
    reported failure upstream.
    """
    from haystack.utils import Secret
    from haystack_integrations.tools.mcp import MCPToolset, StreamableHttpServerInfo

    def _get_toolset() -> MCPToolset:
        server_info = StreamableHttpServerInfo(
            url=url,
            token=Secret.from_token(token) if token else None,
        )
        return MCPToolset(
            server_info=server_info,
            tool_names=tools,
            eager_connect=True,
            connection_timeout=timeout,
        )

    result = list(await asyncio.to_thread(_get_toolset))
    logger.info("Loaded %d tool(s) from %s", len(result), url)
    return result
