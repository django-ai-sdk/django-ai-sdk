"""MCP-backed Integration — connects to a remote MCP server over Streamable HTTP.

Each MCPIntegration owns a ResilientCache (stale-while-revalidate + circuit breaker —
see django_ai_sdk.integrations.base) so discovery (connect + list_tools) never sits
directly on the chat-request critical path.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import httpx
from authlib.integrations.base_client.errors import OAuthError
from authlib.integrations.httpx_client import AsyncOAuth2Client
from django.conf import settings
from pydantic import SecretStr, ValidationError

from django_ai_sdk.integrations.base import (
    IntegrationNotConnectable,
    IntegrationService,
    IntegrationStatus,
    ResilientCache,
)
from django_ai_sdk.integrations.mcp.schemas import (
    OAuthMCPIntegrationConfig,
    StaticMCPIntegrationConfig,
    TokenMCPIntegrationConfig,
    UserTokenMCPIntegrationConfig,
)

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser
    from django.http import HttpRequest

    from django_ai_sdk.assistant import Assistant
    from django_ai_sdk.integrations.mcp.models import MCPOAuthToken

logger = logging.getLogger(__name__)

# Session-key templates for PKCE state during the OAuth redirect dance.
_K_STATE = "mcp_oauth_state_{}"
_K_VERIFIER = "mcp_oauth_verifier_{}"
_K_TOKEN_ENDPOINT = "mcp_oauth_token_endpoint_{}"  # noqa: S105


class MCPIntegration(IntegrationService):
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
        config: StaticMCPIntegrationConfig
        | TokenMCPIntegrationConfig
        | UserTokenMCPIntegrationConfig
        | OAuthMCPIntegrationConfig,
        *,
        needs_setup: str | None = None,
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
        # OAuth: redirect flow. User-token: submit-a-secret flow. Static/token: none.
        self.supports_connect = isinstance(
            config, (OAuthMCPIntegrationConfig, UserTokenMCPIntegrationConfig)
        )
        # A required secret was missing at config-build time (see build_mcp_config_safe)
        # — the integration is registered (so it shows up, e.g. in admin/settings) but
        # reports DISCONNECTED and contributes no tools until configured, rather than
        # crashing app boot.
        self._needs_setup = needs_setup

    @property
    def kind(self) -> str:
        return self.config.type

    @property
    def connect_kind(self) -> str | None:
        if isinstance(self.config, OAuthMCPIntegrationConfig):
            return "oauth"
        if isinstance(self.config, UserTokenMCPIntegrationConfig):
            return "credential"
        return None

    @property
    def detail(self) -> str | None:
        return self._needs_setup

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
        if self._needs_setup:
            return []
        if self.config.tools:
            return self.config.tools
        return [t.name for t in await self.get_tools(user)]

    def _cache_key(self, user: AbstractBaseUser | AnonymousUser | None) -> str | None:
        if isinstance(self.config, (OAuthMCPIntegrationConfig, UserTokenMCPIntegrationConfig)):
            user_id = getattr(user, "pk", None)
            if user_id is None:
                return None
            return f"{self.name}:{user_id}"
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
        if self._needs_setup:
            return []
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
        if self._needs_setup:
            return IntegrationStatus.DISCONNECTED

        if isinstance(self.config, (StaticMCPIntegrationConfig, TokenMCPIntegrationConfig)):
            await self.get_tools(user)
            return self._cache.status_for(self._cache_key(user))

        if isinstance(self.config, UserTokenMCPIntegrationConfig):
            token = await self._get_oauth_token(user)
            if token is None:
                return IntegrationStatus.DISCONNECTED
            await self.get_tools(user)
            return self._cache.status_for(self._cache_key(user))

        token = await self._get_oauth_token(user)
        if token is None:
            return IntegrationStatus.DISCONNECTED
        # An expired access token with no refresh token can't recover without a fresh
        # OAuth flow, so surface EXPIRED to prompt a reconnect. When a refresh token is
        # present, get_tools() below refreshes transparently — fall through and report
        # the real post-refresh outcome rather than a misleading EXPIRED.
        if token.is_expired() and not token.get_refresh_token():
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

    async def disconnect(self, user: AbstractBaseUser | AnonymousUser | None = None) -> bool:
        """Revoke the user's stored OAuth/credential token for this server.

        No-op for static/token (shared-secret) servers — there's no per-user
        connection to drop.
        """
        if user is None or not isinstance(
            self.config, (OAuthMCPIntegrationConfig, UserTokenMCPIntegrationConfig)
        ):
            return False
        from django_ai_sdk.integrations.mcp.models import MCPOAuthToken

        deleted, _ = await MCPOAuthToken.objects.filter(user=user, server_name=self.name).adelete()
        await self.reconnect(user)
        return deleted > 0

    async def store_credential(
        self, user: AbstractBaseUser | AnonymousUser | None, secret: str
    ) -> None:
        """Store a user-submitted token for a ``user_token`` server.

        Stored the same way an OAuth access token is (via ``MCPOAuthToken``), with no
        refresh token or expiry — there's nothing to refresh for a user-supplied secret.
        """
        if not isinstance(self.config, UserTokenMCPIntegrationConfig):
            raise IntegrationNotConnectable(f"{self.name!r} does not accept a stored credential")
        if self._needs_setup:
            raise IntegrationNotConnectable(f"{self.name!r} is not configured: {self._needs_setup}")
        if user is None:
            raise ValueError("store_credential() requires a user")
        if not secret:
            raise ValueError("secret must not be empty")

        from django_ai_sdk.integrations.mcp.models import MCPOAuthToken

        token_obj, _ = await MCPOAuthToken.objects.aget_or_create(user=user, server_name=self.name)
        token_obj.set_tokens({"access_token": secret})
        await token_obj.asave()
        await self.reconnect(user)

    async def refresh(self, user: AbstractBaseUser | AnonymousUser | None = None) -> None:
        """Refresh OAuth tokens (the recurring ``refresh_integrations`` task).

        With a ``user``, refresh that user's token. Without, proactively refresh every
        stored token for this server expiring within
        ``AI_SDK_MCP_REFRESH_THRESHOLD_MINUTES`` (default 10). No-op for non-OAuth
        servers.

        A successful refresh invalidates this user's cached tool list immediately —
        Haystack's ``MCPToolset`` captures the bearer token at construction, so
        without this the already-cached toolset would keep serving the pre-refresh
        (soon-to-expire) token until the tool cache's own TTL happens to roll over,
        rather than picking up the new one right away.
        """
        if not isinstance(self.config, OAuthMCPIntegrationConfig):
            return
        from django.utils import timezone

        from django_ai_sdk.integrations.mcp.models import MCPOAuthToken

        qs = MCPOAuthToken.objects.filter(server_name=self.name)
        if user is not None:
            qs = qs.filter(user=user)
        else:
            threshold = getattr(settings, "AI_SDK_MCP_REFRESH_THRESHOLD_MINUTES", 10)
            qs = qs.filter(expires_at__lte=timezone.now() + timedelta(minutes=threshold))
        async for token_obj in qs:
            refreshed = await refresh_oauth_token(token_obj, self.config)
            if refreshed is not None:
                await self._cache.invalidate(f"{self.name}:{token_obj.user_id}")

    async def connect(
        self,
        user: AbstractBaseUser | AnonymousUser | None = None,
        *,
        request: HttpRequest | None = None,
        redirect_uri: str = "",
    ) -> dict[str, Any]:
        """Begin the OAuth 2.1 + PKCE flow. Stores PKCE state in the session and
        returns ``{"redirect_url": <authorization_url>}`` for the client to follow."""
        if not isinstance(self.config, OAuthMCPIntegrationConfig):
            raise IntegrationNotConnectable(f"{self.name!r} is not an OAuth integration")
        if self._needs_setup:
            raise IntegrationNotConnectable(f"{self.name!r} is not configured: {self._needs_setup}")
        if request is None:
            raise ValueError("connect() for an OAuth integration requires the request")

        from asgiref.sync import sync_to_async

        from django_ai_sdk.integrations.mcp import services as mcp_service

        discovery = await mcp_service.get_oauth_discovery(self.name)
        client_id, _secret = await mcp_service.get_or_register_client(
            self.name, redirect_uri, discovery
        )
        verifier, challenge, state = mcp_service.build_pkce_params()

        request.session[_K_STATE.format(self.name)] = state
        request.session[_K_VERIFIER.format(self.name)] = verifier
        request.session[_K_TOKEN_ENDPOINT.format(self.name)] = discovery.token_endpoint
        await sync_to_async(request.session.save)()

        auth_url = mcp_service.build_auth_url(
            discovery=discovery,
            client_id=client_id,
            redirect_uri=redirect_uri,
            state=state,
            challenge=challenge,
            scope=self.config.scope,
        )
        return {"redirect_url": auth_url}

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
        if isinstance(self.config, UserTokenMCPIntegrationConfig):
            return await self._fetch_user_token(user)
        return await self._fetch_oauth(user)

    async def _fetch_user_token(self, user: AbstractBaseUser | AnonymousUser | None) -> list[Any]:
        assert isinstance(self.config, UserTokenMCPIntegrationConfig)

        token_obj = await self._get_oauth_token(user)
        if token_obj is None:
            return []
        access_token = token_obj.get_access_token()
        if not access_token:
            return []
        return await _connect(
            self.config.url, access_token, self.config.tools or None, timeout=self._timeout
        )

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
        # Only reached via _fetch() for an OAuth config; narrow it for both type-checking
        # and a clear failure if that invariant is ever broken.
        assert isinstance(self.config, OAuthMCPIntegrationConfig)

        token_obj = await self._get_oauth_token(user)
        if token_obj is None:
            # _get_oauth_token() returns None for a None user, so past this point
            # `user` is guaranteed non-None.
            return []
        assert user is not None

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


def build_mcp_config_safe(
    *,
    auth: str,
    url: str,
    label: str,
    tools: list[str],
    scope: str = "",
    client_id: str = "",
    client_secret: str = "",
    oauth_discovery_url: str = "",
    authorization_endpoint: str = "",
    token_endpoint: str = "",
    token: str = "",
) -> tuple[
    StaticMCPIntegrationConfig
    | TokenMCPIntegrationConfig
    | UserTokenMCPIntegrationConfig
    | OAuthMCPIntegrationConfig,
    str | None,
]:
    """Build an MCP config for ``auth``, never raising.

    A misconfigured integration (e.g. ``auth="token"`` with no token set) must not
    crash app boot — the config's own validators (e.g. ``TokenMCPIntegrationConfig``
    requiring a non-empty token) would otherwise raise from deep inside ``ready()``.
    Instead, returns ``(config, needs_setup_reason)``: on success ``needs_setup_reason``
    is ``None``; on failure (missing url, missing required secret, …) it's a
    human-readable reason and ``config`` is a harmless static placeholder that never
    connects (``get_tools``/``get_status`` on the owning ``MCPIntegration`` are
    already short-circuited by ``needs_setup`` before this placeholder is ever used).
    """
    if not url:
        return StaticMCPIntegrationConfig(url="about:blank", label=label, tools=[]), (
            "missing required `url`"
        )
    try:
        if auth == "oauth":
            config = OAuthMCPIntegrationConfig(
                url=url,
                label=label,
                tools=tools,
                scope=scope,
                client_id=client_id,
                client_secret=SecretStr(client_secret),
                oauth_discovery_url=oauth_discovery_url,
                authorization_endpoint=authorization_endpoint,
                token_endpoint=token_endpoint,
            )
        elif auth == "token":
            config = TokenMCPIntegrationConfig(
                url=url, label=label, tools=tools, token=SecretStr(token)
            )
        elif auth == "user_token":
            config = UserTokenMCPIntegrationConfig(url=url, label=label, tools=tools)
        else:
            config = StaticMCPIntegrationConfig(url=url, label=label, tools=tools)
    except ValidationError as e:
        reason = "; ".join(err["msg"] for err in e.errors()) or str(e)
        return StaticMCPIntegrationConfig(url="about:blank", label=label, tools=[]), reason
    return config, None


class MCPIntegrationService(MCPIntegration):
    """Thin base for a known MCP server shipped as its own SDK/product app.

    Subclasses declare the server statically as class attributes and read
    per-deployment params (secrets, tool allow-list, URL overrides) from their
    ``AI_SDK_<NAME>`` settings slice — being in ``INSTALLED_APPS`` is what enables
    them, the slice only feeds credentials/params::

        class NotionService(MCPIntegrationService):
            name = "notion"
            label = "Notion"
            url = "https://mcp.notion.com/mcp"
            auth = "oauth"
            default_tools = ["notion-search"]

    ``AI_SDK_NOTION = {"tools": [...], "client_id": ..., "client_secret": ...}``

    A missing required secret (e.g. ``auth="token"`` with no token configured) never
    crashes boot — the integration registers but reports itself as needing setup
    (``detail`` explains why) until the settings slice is filled in.
    """

    url: str = ""
    auth: str = "static"  # "static" | "token" | "user_token" | "oauth"
    default_tools: list[str] = []
    scope: str = ""
    #: Settings key holding this integration's params; defaults to ``AI_SDK_<NAME>``.
    settings_key: str = ""

    def __init__(self) -> None:
        if not self.name:
            raise ValueError(f"{type(self).__name__} must set a non-empty `name`")
        params = self._get_params()
        config, needs_setup = self._build_config(params)
        super().__init__(self.name, config, needs_setup=needs_setup)
        if needs_setup:
            logger.warning("Integration %r needs setup: %s", self.name, needs_setup)

    def _get_params(self) -> dict[str, Any]:
        key = self.settings_key or f"AI_SDK_{self.name.upper()}"
        return dict(getattr(settings, key, {}) or {})

    def _build_config(
        self, params: dict[str, Any]
    ) -> tuple[
        StaticMCPIntegrationConfig
        | TokenMCPIntegrationConfig
        | UserTokenMCPIntegrationConfig
        | OAuthMCPIntegrationConfig,
        str | None,
    ]:
        return build_mcp_config_safe(
            auth=self.auth,
            url=params.get("url", self.url),
            label=params.get("label") or self.label or self.name.title(),
            tools=list(params.get("tools", self.default_tools)),
            scope=params.get("scope", self.scope),
            client_id=params.get("client_id", ""),
            client_secret=params.get("client_secret", ""),
            oauth_discovery_url=params.get("oauth_discovery_url", ""),
            authorization_endpoint=params.get("authorization_endpoint", ""),
            token_endpoint=params.get("token_endpoint", ""),
            token=params.get("token", ""),
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


def build_oauth_client(client_id: str, client_secret: str, **kwargs: Any) -> AsyncOAuth2Client:
    """Build an Authlib OAuth2 client with the right client-authentication method.

    Confidential clients (a client_secret is configured) authenticate with HTTP
    Basic; public clients send client_id in the request body instead.
    """
    return AsyncOAuth2Client(
        client_id=client_id,
        client_secret=client_secret or None,
        token_endpoint_auth_method="client_secret_basic" if client_secret else "none",
        timeout=10,
        **kwargs,
    )


async def refresh_oauth_token(
    token_obj: MCPOAuthToken, config: OAuthMCPIntegrationConfig
) -> MCPOAuthToken | None:
    """The single refresh_token grant. Returns the updated token_obj, or None on failure.

    Every refresh in the codebase (the loader's own lazy refresh, the OAuth views,
    and the refresh_integrations command) funnels through here.
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

    from django_ai_sdk.integrations.mcp.models import MCPOAuthToken

    # Two callers can redeem the same refresh_token concurrently (a live request
    # racing the refresh_integrations cron, or two overlapping cron runs). Only persist
    # our result if the row still holds the refresh_token we read — otherwise we lost
    # the race and must not clobber the winner's tokens with our own.
    stale_refresh_token = token_obj.refresh_token
    outcome = {"updated": 0}

    async def _persist(token: dict[str, Any], refresh_token: str | None = None, **_: Any) -> None:
        if not token.get("access_token"):
            raise OAuthError(error="invalid_response", description="missing access_token")
        token_obj.set_tokens(token)
        outcome["updated"] = await MCPOAuthToken.objects.filter(
            pk=token_obj.pk, refresh_token=stale_refresh_token
        ).aupdate(
            access_token=token_obj.access_token,
            refresh_token=token_obj.refresh_token,
            token_type=token_obj.token_type,
            scope=token_obj.scope,
            expires_at=token_obj.expires_at,
        )

    client = build_oauth_client(client_id, client_secret, update_token=_persist)
    try:
        async with client:
            await client.refresh_token(discovery.token_endpoint, refresh_token=refresh_token)
    except (httpx.HTTPError, OAuthError) as e:
        # Our own exchange may have failed *because* a concurrent refresh already
        # rotated this refresh_token (e.g. the IDP rejected our now-stale one with
        # invalid_grant). Check whether another writer already landed a good token
        # before reporting failure upstream.
        current = await MCPOAuthToken.objects.aget(pk=token_obj.pk)
        if current.refresh_token != stale_refresh_token and not current.is_expired():
            logger.info(
                "Refresh for %r failed (%s) but another refresh already landed",
                token_obj.server_name,
                e,
            )
            return current
        logger.error("Token refresh failed for server %r: %s", token_obj.server_name, e)
        return None

    if outcome["updated"]:
        logger.info("Refreshed OAuth token for server %r", token_obj.server_name)
        return token_obj

    logger.info(
        "Lost refresh race for %r — another refresh already landed, reloading its result",
        token_obj.server_name,
    )
    return await MCPOAuthToken.objects.aget(pk=token_obj.pk)


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
