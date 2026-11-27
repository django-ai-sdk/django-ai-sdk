from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import SecretStr, ValidationError

from django_ai_sdk.integrations.base import (
    Integration,
    IntegrationNotConnectable,
    IntegrationStatus,
    ResilientCache,
)
from django_ai_sdk.integrations.mcp.schemas import (
    OAuthMCPIntegrationConfig,
    StaticMCPIntegrationConfig,
    TokenMCPIntegrationConfig,
)
from django_ai_sdk.utils import resolve_setting

if TYPE_CHECKING:
    from authlib.integrations.httpx_client import AsyncOAuth2Client
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser
    from django.http import HttpRequest

    from django_ai_sdk.agents.base import Agent
    from django_ai_sdk.integrations.mcp.models import MCPOAuthToken

logger = logging.getLogger(__name__)

# Authlib uses httpx2 when installed, else httpx, so catch both.
try:
    import httpx2

    HTTP_ERRORS: tuple[type[Exception], ...] = (httpx.HTTPError, httpx2.HTTPError)
except ImportError:
    HTTP_ERRORS = (httpx.HTTPError,)

AUTH_KINDS = frozenset({"static", "token", "oauth"})

_MCP_APP = "django_ai_sdk.integrations.mcp"

_K_FLOW = "mcp_oauth_{}_{}"


class DynamicMCPIntegration(Integration):
    def __init__(
        self,
        name: str,
        config: StaticMCPIntegrationConfig | TokenMCPIntegrationConfig | OAuthMCPIntegrationConfig,
        *,
        needs_setup: str | None = None,
        intended_kind: str | None = None,
    ) -> None:
        self.name = name
        self._timeout = resolve_setting("AI_SDK_INTEGRATION_TIMEOUT", 3)
        self._cache = ResilientCache(
            ttl=resolve_setting("AI_SDK_INTEGRATION_CACHE_TTL", 900),
            timeout=self._timeout,
            cb_cooldown=resolve_setting("AI_SDK_INTEGRATION_CB_COOLDOWN", 60),
        )
        self.label = config.label or name.title()
        self.hint = config.hint
        self.config = config
        self.supports_connect = isinstance(config, OAuthMCPIntegrationConfig)
        self.connect_kind = "oauth" if self.supports_connect else None
        self.detail = needs_setup
        self._intended_kind = intended_kind

    @property
    def kind(self) -> str:
        if self.detail and self._intended_kind:
            return self._intended_kind
        return self.config.type

    async def get_tool_names(
        self, user: AbstractBaseUser | AnonymousUser | None = None
    ) -> list[str]:
        """The configured allow-list, else the cached get_tools() names."""
        if self.detail:
            return []
        if self.config.tools:
            return self.config.tools
        return [t.name for t in await self.get_tools(user)]

    def _cache_key(self, user: AbstractBaseUser | AnonymousUser | None) -> str | None:
        if isinstance(self.config, OAuthMCPIntegrationConfig):
            user_id = getattr(user, "pk", None)
            if user_id is None:
                return None
            return f"{self.name}:{user_id}"
        return self.name

    async def get_tools(
        self,
        user: AbstractBaseUser | AnonymousUser | None = None,
        agent: Agent | None = None,
        thread_id: str = "",
    ) -> list[Any]:
        if self.detail:
            return []
        key = self._cache_key(user)
        if key is None:
            return []
        return await self._cache.get(key, lambda: self._fetch(user))

    async def get_status(
        self,
        user: AbstractBaseUser | AnonymousUser | None = None,
        agent: Agent | None = None,
    ) -> IntegrationStatus:
        """Status of a real attempt: ACTIVE only if the last fetch succeeded."""
        if self.detail:
            return IntegrationStatus.DISCONNECTED

        if isinstance(self.config, (StaticMCPIntegrationConfig, TokenMCPIntegrationConfig)):
            await self.get_tools(user)
            return self._cache.status_for(self._cache_key(user))

        token = await self._get_oauth_token(user)
        if token is None:
            return IntegrationStatus.DISCONNECTED
        if token.is_expired() and not token.get_refresh_token():
            return IntegrationStatus.EXPIRED

        await self.get_tools(user)
        return self._cache.status_for(self._cache_key(user))

    async def reconnect(self, user: AbstractBaseUser | AnonymousUser | None = None) -> None:
        key = self._cache_key(user)
        if key is not None:
            await self._cache.invalidate(key)

    async def disconnect(self, user: AbstractBaseUser | AnonymousUser | None = None) -> bool:
        """Delete the user's stored OAuth token; no-op for static/token servers."""
        if user is None or not isinstance(self.config, OAuthMCPIntegrationConfig):
            return False
        from django_ai_sdk.integrations.mcp.models import MCPOAuthToken

        deleted, _ = await MCPOAuthToken.objects.filter(user=user, server_name=self.name).adelete()
        await self.reconnect(user)
        return deleted > 0

    async def refresh(self, user: AbstractBaseUser | AnonymousUser | None = None) -> None:
        """Refresh the user's token, or without a user every token expiring soon."""
        if not isinstance(self.config, OAuthMCPIntegrationConfig):
            return
        from django.utils import timezone

        from django_ai_sdk.integrations.mcp.models import MCPOAuthToken

        qs = MCPOAuthToken.objects.filter(server_name=self.name)
        if user is not None:
            qs = qs.filter(user=user)
        else:
            threshold = resolve_setting("AI_SDK_MCP_REFRESH_THRESHOLD_MINUTES", 10)
            qs = qs.filter(expires_at__lte=timezone.now() + timedelta(minutes=threshold))
        async for token_obj in qs:
            refreshed = await refresh_oauth_token(token_obj, self.config)
            if refreshed is not None:
                # MCPToolset captures the bearer token, so drop the cached toolset.
                await self._cache.invalidate(f"{self.name}:{token_obj.user_id}")

    async def connect(
        self,
        user: AbstractBaseUser | AnonymousUser | None = None,
        *,
        request: HttpRequest | None = None,
        redirect_uri: str = "",
    ) -> dict[str, Any]:
        """Begin the OAuth 2.1 + PKCE flow; returns {"redirect_url": ...}."""
        if not isinstance(self.config, OAuthMCPIntegrationConfig):
            raise IntegrationNotConnectable(f"{self.name!r} is not an OAuth integration")
        if self.detail:
            raise IntegrationNotConnectable(f"{self.name!r} is not configured: {self.detail}")
        if request is None:
            raise ValueError("connect() for an OAuth integration requires the request")

        from asgiref.sync import sync_to_async

        from django_ai_sdk.integrations.mcp import services as mcp_service

        discovery = await mcp_service.get_oauth_discovery(self.name)
        client_id, _secret = await mcp_service.get_or_register_client(
            self.name, redirect_uri, discovery
        )
        verifier, challenge, state = mcp_service.build_pkce_params()

        request.session[_K_FLOW.format(self.name, state)] = {
            "verifier": verifier,
            "token_endpoint": discovery.token_endpoint,
        }
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
        assert isinstance(self.config, OAuthMCPIntegrationConfig)

        token_obj = await self._get_oauth_token(user)
        if token_obj is None:
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
    hint: str = "",
    scope: str = "",
    client_id: str = "",
    client_secret: str = "",
    oauth_discovery_url: str = "",
    authorization_endpoint: str = "",
    token_endpoint: str = "",
    token: str = "",
) -> tuple[
    StaticMCPIntegrationConfig | TokenMCPIntegrationConfig | OAuthMCPIntegrationConfig,
    str | None,
]:
    if not url:
        return StaticMCPIntegrationConfig(url="about:blank", label=label, tools=[]), (
            "missing required url"
        )
    if auth not in AUTH_KINDS:
        return StaticMCPIntegrationConfig(url="about:blank", label=label, tools=[]), (
            f"unknown auth {auth!r}; expected one of {', '.join(sorted(AUTH_KINDS))}"
        )
    try:
        if auth == "oauth":
            config = OAuthMCPIntegrationConfig(
                url=url,
                label=label,
                hint=hint,
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
                url=url, label=label, hint=hint, tools=tools, token=SecretStr(token)
            )
        else:
            config = StaticMCPIntegrationConfig(url=url, label=label, hint=hint, tools=tools)
    except ValidationError as e:
        reason = "; ".join(err["msg"] for err in e.errors()) or str(e)
        return StaticMCPIntegrationConfig(url="about:blank", label=label, tools=[]), reason
    return config, None


class MCPIntegration(DynamicMCPIntegration):
    """Base for a known MCP server app; settings override AUTH, URL, LABEL, TOOLS, SCOPE."""

    url: str = ""
    auth: str = "static"
    default_tools: list[str] = []
    scope: str = ""
    hint: str = ""

    CONFIG_KEYS = frozenset(
        {
            "AUTH",
            "URL",
            "LABEL",
            "TOOLS",
            "SCOPE",
            "TOKEN",
            "CLIENT_ID",
            "CLIENT_SECRET",
            "OAUTH_DISCOVERY_URL",
            "AUTHORIZATION_ENDPOINT",
            "TOKEN_ENDPOINT",
        }
    )

    def __init__(self) -> None:
        if not self.name:
            raise ValueError(f"{type(self).__name__} must set a non-empty `name`")
        self._warn_unknown_config_keys()
        auth = self._auth()
        config, needs_setup = self._build_config()
        if not needs_setup:
            needs_setup = self._missing_mcp_app()
        super().__init__(self.name, config, needs_setup=needs_setup, intended_kind=auth)
        if needs_setup:
            logger.warning("Integration %r needs setup: %s", self.name, needs_setup)

    def _auth(self) -> str:
        return str(self._config().get("AUTH") or self.auth)

    def _warn_unknown_config_keys(self) -> None:
        unknown = set(self._config()) - self.CONFIG_KEYS
        if unknown:
            logger.warning(
                "AI_SDK_INTEGRATIONS[%r] has unrecognised keys %s; expected any of %s",
                self.name,
                ", ".join(sorted(unknown)),
                ", ".join(sorted(self.CONFIG_KEYS)),
            )

    def _missing_mcp_app(self) -> str | None:
        from django.apps import apps as django_apps

        if django_apps.is_installed(_MCP_APP):
            return None
        return f"add {_MCP_APP!r} to INSTALLED_APPS; MCP integrations need its models"

    def _build_config(
        self,
    ) -> tuple[
        StaticMCPIntegrationConfig | TokenMCPIntegrationConfig | OAuthMCPIntegrationConfig,
        str | None,
    ]:
        config = self._config()
        return build_mcp_config_safe(
            auth=self._auth(),
            url=config.get("URL") or self.url,
            label=config.get("LABEL") or self.label or self.name.title(),
            hint=self.hint,
            tools=list(config.get("TOOLS") or self.default_tools),
            scope=config.get("SCOPE") or self.scope,
            client_id=self.secret("client_id"),
            client_secret=self.secret("client_secret"),
            oauth_discovery_url=self.secret("oauth_discovery_url"),
            authorization_endpoint=self.secret("authorization_endpoint"),
            token_endpoint=self.secret("token_endpoint"),
            token=self.secret("token"),
        )


async def resolve_client_credentials(
    server_name: str, config: OAuthMCPIntegrationConfig
) -> tuple[str, str]:
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
    from authlib.integrations.httpx_client import AsyncOAuth2Client

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
    refresh_token = token_obj.get_refresh_token()
    if not refresh_token:
        logger.warning("No refresh token available for server %r", token_obj.server_name)
        return None

    from authlib.integrations.base_client.errors import OAuthError

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
    except (*HTTP_ERRORS, OAuthError) as e:
        current = await MCPOAuthToken.objects.filter(pk=token_obj.pk).afirst()
        if (
            current is not None
            and current.refresh_token != stale_refresh_token
            and not current.is_expired()
        ):
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
    """timeout goes to MCPToolset itself: the asyncio timeout can't interrupt its thread."""
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
