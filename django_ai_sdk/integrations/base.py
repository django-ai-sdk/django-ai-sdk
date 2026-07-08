"""Shared contract and caching for integrations.

An ``Integration`` wraps something an assistant plugs in — an MCP server, a
hand-written API wrapper, etc. ``ResilientCache`` caches a backend's live-fetched
data per key, with stale-while-revalidate refresh and a circuit breaker for repeated
failures. The cache/breaker mechanics are provided by ``cashews`` (a maintained async
cache library); this module is a thin facade that keeps a small ``get/status_for/
invalidate/warm`` surface and adds the per-key health tracking the UI reads.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from cashews import Cache, CircuitBreakerOpen

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.assistant import Assistant

logger = logging.getLogger(__name__)


class IntegrationStatus(StrEnum):
    """An integration's current health.

    ``ResilientCache.status_for()`` produces ACTIVE/DEGRADED — the circuit-breaker
    states. A DEGRADED integration keeps retrying automatically: after a cooldown the
    breaker half-opens, probes once, and closes again on success (standard half-open
    recovery). EXPIRED/DISCONNECTED describe credential state instead (e.g. a lapsed
    OAuth token) and are decided by the Integration subclass itself, before it even
    touches the cache.
    """

    ACTIVE = "active"
    DEGRADED = "degraded"  # recent failures; auto-retries, half-opens after a cooldown
    EXPIRED = "expired"  # was connected, but the credential (e.g. OAuth) has lapsed
    DISCONNECTED = "disconnected"  # never connected yet


class Integration(ABC):
    """Common interface every integration backend implements."""

    name: str = ""
    label: str = ""

    @abstractmethod
    async def get_tools(
        self,
        user: AbstractBaseUser | AnonymousUser | None = None,
        assistant: Assistant | None = None,
        thread_id: str = "",
    ) -> list[Any]:
        """Return this integration's tool objects.

        ``assistant`` is the calling Assistant instance. ``thread_id`` is the
        active conversation, forwarded from ``Assistant.get_tools()`` — most
        integrations (MCP servers, external APIs) are user/assistant-scoped and
        can ignore it; it exists for the rare integration whose tools need to
        know which thread they're running in (e.g. one that lists documents
        attached to *this* conversation).
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

        Derives the names from ``get_tools()`` by default, which may do I/O (e.g. an
        MCP connect or an API health check). Callers that assume this is cheap should
        confirm the concrete Integration overrides it with a cached/static source;
        a custom subclass that doesn't override this will pay that I/O cost too.
        """
        return [t.name for t in await self.get_tools(user)]

    async def reconnect(self, user: AbstractBaseUser | AnonymousUser | None = None) -> None:
        """Reset this integration to a fresh state. No-op by default."""


# Standard circuit-breaker defaults — open once a clear majority of a small sample of
# attempts fail, then half-open probe after the cooldown. Not exposed as settings: the
# whole point of adopting a library breaker is to stop hand-tuning bespoke thresholds.
_CB_ERRORS_RATE = 50  # percent of calls-in-window that must fail to trip
_CB_MIN_CALLS = 3  # need at least this many attempts before the breaker can trip
_CB_PERIOD = 600  # seconds; window over which the error rate is measured


class ResilientCache:
    """Stale-while-revalidate cache with a per-key circuit breaker, backed by cashews.

    - Fresh hit: return the cached value.
    - Stale hit (past ``early_ttl``): return the cached value and refresh in the
      background (cashews ``early``).
    - Miss: fetch with a timeout; on failure, record it and return ``empty()``.
    - After repeated failures for a key the breaker opens and fetches are skipped
      (``empty()`` served) for a cooldown, then it half-opens and probes once —
      success closes it, failure re-opens it. There is no terminal "give up" state;
      recovery is automatic.

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
        # Per-key record of whether the last attempt succeeded, for status_for(). The
        # cashews breaker deliberately exposes no state-query API, so health for the UI
        # is tracked here rather than read out of the breaker.
        self._last_ok: dict[str, bool] = {}
        self._cache = Cache()
        self._cache.setup("mem://")
        # Refresh in the background once the entry is 80% of the way to expiry, so a
        # fresh value is usually ready before the hard TTL — kept as a float so it
        # stays below `ttl` for sub-second TTLs too (used in tests).
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
            # Raises on failure/timeout so the breaker counts it and nothing is cached;
            # the value is only cached (and served stale) once a fetch succeeds.
            return await asyncio.wait_for(fetch(), timeout=self._timeout)

        self._run = _run

    @staticmethod
    def _norm(key: Any) -> str:
        return str(key)

    async def get(self, key: Any, fetch: Callable[[], Awaitable[Any]]) -> Any:
        """Return the cached value for ``key``, per the rules in the class docstring.

        ``fetch`` is an async, no-arg callable performing the live I/O. Never raises —
        a failed/timed-out/short-circuited fetch is recorded and ``empty()`` returned.
        """
        k = self._norm(key)
        try:
            value = await self._run(k, fetch)
        except CircuitBreakerOpen:
            self._last_ok[k] = False
            return self._empty()
        except Exception:
            logger.warning("Integration fetch failed/timed out for %r", key, exc_info=True)
            self._last_ok[k] = False
            return self._empty()
        self._last_ok[k] = True
        return value

    def status_for(self, key: Any) -> IntegrationStatus:
        """Return the status based on the last attempt for ``key``.

        A key that was never attempted reads as ACTIVE — there's no "unknown" state.
        Callers MUST call ``get()`` (or ``warm()``) for ``key`` first, or this reports
        a false ACTIVE for an integration nothing has actually checked yet.
        """
        ok = self._last_ok.get(self._norm(key))
        if ok is None or ok:
            return IntegrationStatus.ACTIVE
        return IntegrationStatus.DEGRADED

    async def invalidate(self, key: Any) -> None:
        """Drop the cached value and reset the breaker for ``key`` — a "retry now".

        cashews stores these under its own decorator prefixes plus an internal format
        version (e.g. ``early:v2:v:{k}``), so the value match uses a leading wildcard
        rather than hard-coding that version marker; the breaker keys live under the
        stable ``circuit_breaker:`` prefix.
        """
        k = self._norm(key)
        await self._cache.delete_match(f"*:v:{k}")
        await self._cache.delete_match(f"circuit_breaker:cb:{k}:*")
        self._last_ok.pop(k, None)

    async def warm(self, key: Any, fetch: Callable[[], Awaitable[Any]]) -> None:
        """Populate the cache for ``key`` if it isn't already cached.

        Fetch failures are already handled by ``get()``, which records them and
        returns ``empty()`` instead of raising.
        """
        await self.get(key, fetch)
