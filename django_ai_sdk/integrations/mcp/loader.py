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

from django_ai_sdk.integrations.base import Integration, IntegrationStatus, ResilientCache
from django_ai_sdk.integrations.mcp.schemas import (
    OAuthMCPIntegrationConfig,
    StaticMCPIntegrationConfig,
    TokenMCPIntegrationConfig,
)

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

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
            cb_threshold=getattr(settings, "AI_SDK_INTEGRATION_CB_THRESHOLD", 3),
            cb_cooldown=getattr(settings, "AI_SDK_INTEGRATION_CB_COOLDOWN", 60),
            cb_max_cooldown=getattr(settings, "AI_SDK_INTEGRATION_CB_MAX_COOLDOWN", 1800),
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
    ) -> list[Any]:
        # `assistant` is unused here — MCP tools are discovered from the remote
        # server as-is, with no per-call model/context to inject. Accepted only for
        # signature consistency with the Integration ABC.
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
            return self._cache.status_for(self.name)

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
        """Manually reset this integration back to a fresh, never-attempted state.

        The only way out of BROKEN — auto-retry stops there by design (see
        ResilientCache), so something (a staff action, a management command, a
        "Reconnect" button) has to call this once the actual problem (wrong URL,
        bad token, etc.) has been fixed.
        """
        key = self._cache_key(user)
        if key is not None:
            self._cache.invalidate(key)

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


async def refresh_oauth_token(
    token_obj: MCPOAuthToken, config: OAuthMCPIntegrationConfig
) -> MCPOAuthToken | None:
    """Attempt a refresh_token grant. Returns the updated token_obj or None on failure."""
    refresh_token = token_obj.get_refresh_token()
    if not refresh_token:
        logger.warning("No refresh token available for server %r", token_obj.server_name)
        return None

    from django_ai_sdk.integrations.mcp.discovery import discover
    from django_ai_sdk.integrations.mcp.models import MCPOAuthClient

    try:
        discovery = await discover(config.url)
    except Exception as e:
        logger.error(
            "Cannot discover token endpoint for refresh of %r: %s", token_obj.server_name, e
        )
        return None

    client_id = config.client_id
    client_secret = config.client_secret.get_secret_value()
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
