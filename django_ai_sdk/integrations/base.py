"""Shared contract and caching for integrations.

Integration wraps something an assistant plugs in, such as an MCP server or a
hand-written API wrapper. ResilientCache caches a backend's live-fetched data per key,
with stale-while-revalidate refresh and a circuit breaker for repeated failures. The
cache and breaker mechanics come from cashews; this module adds a small
get/status_for/invalidate surface plus the per-key health tracking the UI reads.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from cashews import Cache, CircuitBreakerOpen

from django_ai_sdk.permissions import Operation, PermissionDomain

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.assistant import Assistant
    from django_ai_sdk.permissions import BasePermission

logger = logging.getLogger(__name__)


class IntegrationNotConnectable(Exception):
    """Raised when connect() is called on an integration that doesn't support it."""


class IntegrationStatus(StrEnum):
    """An integration's current health.

    ResilientCache.status_for() produces ACTIVE/DEGRADED, the circuit-breaker states.
    A DEGRADED integration keeps retrying automatically: after a cooldown the breaker
    half-opens, probes once, and closes again on success. EXPIRED/DISCONNECTED
    describe credential state instead (e.g. a lapsed OAuth token) and are decided by
    the Integration subclass itself, before it touches the cache.
    """

    ACTIVE = "active"
    DEGRADED = "degraded"  # recent failures; auto-retries, half-opens after a cooldown
    EXPIRED = "expired"  # was connected, but the credential (e.g. OAuth) has lapsed
    DISCONNECTED = "disconnected"  # never connected yet


class Integration(ABC):
    """The single point of contact for one integration.

    Every integration, whether an MCP server or a hand-written API wrapper, is one
    Integration subclass, living in its app's integration.py. It owns the
    integration's tools, health, permissions, connection lifecycle and credential
    refresh, and is what the Assistant and the host project's integrations endpoints
    talk to. An IntegrationAppConfig (see apps.py) constructs it and registers it
    into the process registry on app ready().
    """

    name: str = ""
    label: str = ""

    #: Permission classes gating this integration (like Assistant.permissions).
    #: Empty falls back to the INTEGRATIONS domain default (see permissions.py).
    permissions: list[type[BasePermission] | BasePermission] = []
    domain: PermissionDomain = PermissionDomain.INTEGRATIONS

    #: Connection-management capabilities the generic /api/integrations router
    #: reads to decide which actions to offer. The router dispatches to the methods
    #: below rather than branching on kind, and lets the service decide.
    supports_connect: bool = False
    supports_test: bool = True

    #: How the client should let a user connect, when supports_connect is True.
    #: "oauth" (call POST /{name}/connect, then follow its redirect_url) is the only
    #: kind today; it is a string rather than a bool so a second flow (e.g. submitting
    #: a per-user token) can be added later without changing the contract.
    connect_kind: str | None = None

    #: Human-readable reason this integration can't run yet (e.g. a missing secret),
    #: or None when fully configured. Never raises; surfaced to the UI instead.
    detail: str | None = None

    # -- configuration --

    def secret(self, key: str, default: str = "") -> str:
        """This integration's configured string value for `key`.

        Reads AI_SDK_INTEGRATIONS[self.name][KEY] (see config.py), so an integration
        named "zendesk" reads AI_SDK_INTEGRATIONS["zendesk"]["TOKEN"]. Returns
        ``default`` when unset -- check it and set `detail` rather than raising, so a
        missing credential degrades this integration instead of failing app boot.
        """
        value = self._config().get(key.upper())
        if value is None:
            return default
        if not isinstance(value, str):
            # Stringifying would turn a list into "['a']" and hand that to a remote
            # server as a credential. Name the key, never the value -- this is a secret.
            logger.warning(
                "AI_SDK_INTEGRATIONS[%r][%r] must be a string, got %s; treating it as unset",
                self.name,
                key.upper(),
                type(value).__name__,
            )
            return default
        return value or default

    def _config(self) -> dict[str, Any]:
        """This integration's AI_SDK_INTEGRATIONS slice, with upper-cased keys.

        Internal: subclasses read individual values through secret(), and MCPIntegration
        reads its non-secret keys (URL, TOOLS, ...) here. Empty when the integration
        isn't configured; never raises.
        """
        from django_ai_sdk.integrations.config import get_integration_config

        return get_integration_config(self.name)

    @abstractmethod
    async def get_tools(
        self,
        user: AbstractBaseUser | AnonymousUser | None = None,
        assistant: Assistant | None = None,
        thread_id: str = "",
    ) -> list[Any]:
        """Return this integration's tool objects.

        assistant is the calling Assistant instance. thread_id is the active
        conversation, forwarded from Assistant.get_tools(). Most integrations (MCP
        servers, external APIs) are user/assistant-scoped and can ignore it; it
        exists for the rare integration whose tools need to know which thread
        they're running in, e.g. one that lists documents attached to that
        conversation.
        """

    @abstractmethod
    async def get_status(
        self,
        user: AbstractBaseUser | AnonymousUser | None = None,
        assistant: Assistant | None = None,
    ) -> IntegrationStatus:
        """Return this integration's current health."""

    @property
    def kind(self) -> str:
        """Category label for display purposes. Defaults to "api"."""
        return "api"

    async def get_tool_names(
        self, user: AbstractBaseUser | AnonymousUser | None = None
    ) -> list[str]:
        """Return this integration's tool names, for display purposes.

        Derives the names from get_tools() by default, which may do I/O (e.g. an
        MCP connect or an API health check). Callers that assume this is cheap should
        confirm the concrete service overrides it with a cached/static source; a
        custom subclass that doesn't override this will pay that I/O cost too.
        """
        return [t.name for t in await self.get_tools(user)]

    async def reconnect(self, user: AbstractBaseUser | AnonymousUser | None = None) -> None:
        """Reset this integration to a fresh state. No-op by default."""

    # -- connection lifecycle (polymorphic; the router calls these, never on `kind`) --

    async def connect(
        self,
        user: AbstractBaseUser | AnonymousUser | None = None,
        *,
        request: Any = None,
        redirect_uri: str = "",
    ) -> dict[str, Any]:
        """Begin connecting this integration for user.

        Return a dict the client acts on, e.g. {"redirect_url": ...} for an OAuth
        flow. request is passed through for services (like MCP-OAuth) that need
        session/PKCE state. Integrations that need no connection step raise
        IntegrationNotConnectable (guarded by supports_connect).
        """
        raise IntegrationNotConnectable(f"{self.name!r} does not support connect()")

    async def disconnect(self, user: AbstractBaseUser | AnonymousUser | None = None) -> bool:
        """Drop any stored credential/connection for user. No-op default."""
        return False

    async def test(self, user: AbstractBaseUser | AnonymousUser | None = None) -> IntegrationStatus:
        """Force a fresh connection attempt and report the real outcome.

        Default: reset cached state and re-probe via get_status().
        """
        await self.reconnect(user)
        return await self.get_status(user)

    # -- lifecycle: refresh (recurring) --

    async def refresh(self, user: AbstractBaseUser | AnonymousUser | None = None) -> None:
        """Refresh credentials (e.g. rotate an OAuth token). Recurring task. No-op default.

        Caches populate lazily on first use through ResilientCache's
        stale-while-revalidate behavior, so there is no boot-time warmup hook.
        Deployers who want pre-warming can call get_status() at deploy time instead.
        """

    # -- permissions --

    async def has_perms(
        self,
        user: AbstractBaseUser | AnonymousUser | None,
        operation: Operation = Operation.USE_INTEGRATION,
        *,
        raise_on_deny: bool = False,
    ) -> bool:
        """Whether user may use/manage this integration.

        Resolves self.permissions (falling back to the INTEGRATIONS domain default)
        and checks them.
        """
        from django_ai_sdk.permissions import get_integration_permissions
        from django_ai_sdk.permissions import has_perms as _has_perms

        return await _has_perms(
            user,
            operation,
            permissions=get_integration_permissions(self),
            raise_on_deny=raise_on_deny,
        )


# Standard circuit-breaker defaults: open once a clear majority of a small sample of
# attempts fail, then half-open probe after the cooldown. Not exposed as settings, to
# avoid hand-tuning bespoke thresholds per integration.
_CB_ERRORS_RATE = 50  # percent of calls-in-window that must fail to trip
_CB_MIN_CALLS = 3  # need at least this many attempts before the breaker can trip
_CB_PERIOD = 600  # seconds; window over which the error rate is measured


class ResilientCache:
    """Stale-while-revalidate cache with a per-key circuit breaker, backed by cashews.

    - Fresh hit: return the cached value.
    - Stale hit (past early_ttl): return the cached value and refresh in the
      background (cashews' "early" mechanism).
    - Miss: fetch with a timeout; on failure, record it and return empty().
    - After repeated failures for a key the breaker opens and fetches are skipped
      (empty() served) for a cooldown, then it half-opens and probes once: success
      closes it, failure re-opens it. There is no terminal give-up state; recovery
      is automatic.

    Each key is isolated: one dead server's breaker never trips another's. Every
    instance owns its own in-process cashews backend, so instances (e.g. in tests)
    don't share state.
    """

    def __init__(
        self,
        *,
        ttl: float,
        timeout: float,
        cb_cooldown: float = 60,
        empty: Callable[[], Any] = list,
    ) -> None:
        self._timeout = timeout
        self._empty = empty
        self._cb_cooldown = cb_cooldown
        # Per-key record of whether the last attempt succeeded, for status_for().
        # cashews exposes no breaker state-query API, so health for the UI is
        # tracked here instead.
        self._last_ok: dict[str, bool] = {}
        # A real fetch failure also sets _last_ok[k] = False, so _last_ok alone
        # can't distinguish "just failed" from "breaker already open" (both read as
        # False). This set tracks the breaker specifically, for the open-transition
        # log below.
        self._breaker_open: set[str] = set()
        self._cache = Cache()
        self._cache.setup("mem://")
        # Refresh in the background once the entry is 80% of the way to expiry, so a
        # fresh value is usually ready before the hard TTL. Kept as a float so it
        # stays below ttl for sub-second TTLs too (used in tests).
        early_ttl = ttl * 0.8

        @self._cache.circuit_breaker(
            errors_rate=_CB_ERRORS_RATE,
            period=_CB_PERIOD,
            ttl=cb_cooldown,
            half_open_ttl=cb_cooldown,
            min_calls=_CB_MIN_CALLS,
            key="cb:{key}",
        )
        @self._cache.early(ttl=ttl, early_ttl=early_ttl, key="v:{key}")
        async def _run(key: str, fetch: Callable[[], Awaitable[Any]]) -> Any:
            # Raises on failure/timeout so the breaker counts it and nothing is
            # cached; the value is only cached (and served stale) once a fetch
            # succeeds.
            return await asyncio.wait_for(fetch(), timeout=self._timeout)

        self._run = _run

    @staticmethod
    def _norm(key: Any) -> str:
        return str(key)

    async def get(self, key: Any, fetch: Callable[[], Awaitable[Any]]) -> Any:
        """Return the cached value for key, per the rules in the class docstring.

        fetch is an async, no-arg callable performing the live I/O. Never raises: a
        failed, timed-out, or short-circuited fetch is recorded and empty() returned.

        Logs only on state transitions (closed to open, open to closed), not per
        call. While a breaker is open, every subsequent call raises
        CircuitBreakerOpen without touching fetch, so logging every one of those
        would just spam. Logging only transitions still means an operator sees when
        an integration goes down and when it recovers, instead of the two being
        indistinguishable from total silence.
        """
        k = self._norm(key)
        was_ok = self._last_ok.get(k)
        try:
            value = await self._run(k, fetch)
        except CircuitBreakerOpen:
            self._last_ok[k] = False
            if k not in self._breaker_open:
                self._breaker_open.add(k)
                logger.warning(
                    "Integration %r: circuit breaker just opened after repeated "
                    "failures — serving empty results until the %.0fs cooldown "
                    "elapses, then it will probe once and recover automatically.",
                    key,
                    self._cb_cooldown,
                )
            return self._empty()
        except Exception:
            logger.warning("Integration fetch failed/timed out for %r", key, exc_info=True)
            self._last_ok[k] = False
            self._breaker_open.discard(k)
            return self._empty()
        self._last_ok[k] = True
        if was_ok is False or k in self._breaker_open:
            logger.info("Integration %r: recovered — fetch succeeded again.", key)
        self._breaker_open.discard(k)
        return value

    def status_for(self, key: Any) -> IntegrationStatus:
        """Return the status based on the last attempt for key.

        Callers must call get() for key first; DynamicMCPIntegration.get_status and
        APIIntegration.get_status both do that before calling this. This raises
        rather than guessing, so a subclass that skips the get() call fails loudly in
        development instead of silently reporting a false ACTIVE for an integration
        nothing has actually checked yet.
        """
        k = self._norm(key)
        ok = self._last_ok.get(k)
        if ok is None:
            raise RuntimeError(
                f"status_for({key!r}) called before get({key!r}) ever ran for this "
                "cache — there's no attempt to report a status for. Call get() first "
                "(get_tools()/health_check for the built-in integrations already do)."
            )
        return IntegrationStatus.ACTIVE if ok else IntegrationStatus.DEGRADED

    async def invalidate(self, key: Any) -> None:
        """Drop cached values and reset the breaker, i.e. force a retry now.

        Clears this instance's whole backend rather than matching key's entries.
        Every instance owns a private in-process backend holding one integration's
        keys, so the blast radius is that one integration: with a per-user key
        (OAuth), other users lose a cached tool list and pay one bounded re-fetch.
        That is a fair price for not depending on cashews' internal key layout;
        matching precisely would mean reproducing its decorator prefixes and an
        internal format version, which is the kind of coupling that breaks silently
        on a library upgrade.

        key is still taken, and its health entry cleared, so callers read naturally
        and a future precise implementation needs no signature change.
        """
        await self._cache.clear()
        self._last_ok.pop(self._norm(key), None)
        self._breaker_open.discard(self._norm(key))
