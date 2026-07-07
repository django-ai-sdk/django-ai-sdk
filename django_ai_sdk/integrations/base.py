"""Shared contract and caching for integrations.

An ``Integration`` wraps something an assistant plugs in — an MCP server, a
hand-written API wrapper, etc. ``ResilientCache`` caches a backend's live-fetched
data per key, with stale-while-revalidate refresh and a circuit breaker for repeated
failures.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.assistant import Assistant

logger = logging.getLogger(__name__)


class IntegrationStatus(StrEnum):
    """An integration's current health.

    ``ResilientCache.status_for()`` only ever produces ACTIVE/DEGRADED/BROKEN — the
    circuit-breaker states. EXPIRED/DISCONNECTED describe credential state instead
    (e.g. a lapsed OAuth token) and are decided by the Integration subclass itself,
    before it even touches the cache.
    """

    ACTIVE = "active"
    DEGRADED = "degraded"  # recent failures, still retrying automatically
    BROKEN = "broken"  # stopped retrying automatically, needs a manual reconnect
    EXPIRED = "expired"  # was connected, but the credential (e.g. OAuth) has lapsed
    DISCONNECTED = "disconnected"  # never connected yet


class Integration(ABC):
    """Common interface every integration backend implements."""

    name: str
    label: str

    @abstractmethod
    async def get_tools(
        self,
        user: AbstractBaseUser | AnonymousUser | None = None,
        assistant: Assistant | None = None,
    ) -> list[Any]:
        """Return this integration's tool objects.

        ``assistant`` is the calling Assistant instance.
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

        Derives the names from ``get_tools()`` by default. Subclasses can override
        with a cheaper source.
        """
        return [t.name for t in await self.get_tools(user)]

    async def reconnect(self, user: AbstractBaseUser | AnonymousUser | None = None) -> None:
        """Reset this integration to a fresh state. No-op by default."""


class _CacheEntry:
    __slots__ = ("value", "expires_at")

    def __init__(self, value: Any, expires_at: float) -> None:
        self.value = value
        self.expires_at = expires_at


class ResilientCache:
    """Stale-while-revalidate cache with a per-key circuit breaker.

    - Fresh cache hit: return the cached value.
    - Stale cache hit: return the cached value, and refresh it in the background.
    - Cache miss: fetch with a timeout; on failure, record it and return ``empty()``.
    - After ``cb_threshold`` consecutive failures for a key, live fetches are skipped
      for a cooldown window and the cached/empty value is served instead. Each time a
      post-cooldown probe fails again, the cooldown doubles, starting from
      ``cb_cooldown`` and capped at ``cb_max_cooldown``.
    - Once a key has sat through one full cooldown at the cap and fails again, it is
      marked BROKEN and stops auto-retrying. ``invalidate()`` resets it.

    All state is in-memory, per process.
    """

    def __init__(
        self,
        *,
        ttl: float,
        timeout: float,
        cb_threshold: int,
        cb_cooldown: float,
        cb_max_cooldown: float,
        empty: Callable[[], Any] = list,
    ) -> None:
        self._ttl = ttl
        self._timeout = timeout
        self._cb_threshold = cb_threshold
        self._cb_cooldown = cb_cooldown
        self._cb_max_cooldown = cb_max_cooldown
        self._empty = empty
        self._entries: dict[Any, _CacheEntry] = {}
        self._locks: dict[Any, asyncio.Lock] = {}
        # Keys with a background refresh in flight.
        self._refreshing: set[Any] = set()
        # Consecutive failure count per key.
        self._failures: dict[Any, int] = {}
        # Monotonic deadline per key; live fetches are skipped until then.
        self._circuit_open_until: dict[Any, float] = {}
        # Doublings applied so far: cooldown = cb_cooldown * 2**level.
        self._backoff_level: dict[Any, int] = {}
        # Keys whose cooldown already hit cb_max_cooldown once; their next failure gives up.
        self._backoff_maxed: set[Any] = set()
        # Keys that gave up auto-retrying. Only invalidate() clears this.
        self._broken: set[Any] = set()
        # Outcome of the most recent attempt per key, for status_for().
        self._last_ok: dict[Any, bool] = {}

    def _lock_for(self, key: Any) -> asyncio.Lock:
        return self._locks.setdefault(key, asyncio.Lock())

    def _circuit_open(self, key: Any) -> bool:
        return key in self._broken or self._circuit_open_until.get(key, 0.0) > time.monotonic()

    def status_for(self, key: Any) -> IntegrationStatus:
        """Return the status based on the last attempt for ``key``.

        A key that was never attempted reads as ACTIVE — there's no "unknown" state.
        Callers MUST call ``get()`` (or ``warm()``) for ``key`` first, or this reports
        a false ACTIVE for an integration nothing has actually checked yet.
        """
        if key in self._broken:
            return IntegrationStatus.BROKEN
        if self._circuit_open(key):
            return IntegrationStatus.DEGRADED
        if self._last_ok.get(key) is False:
            return IntegrationStatus.DEGRADED
        return IntegrationStatus.ACTIVE

    def _clear_circuit_state(self, key: Any) -> None:
        self._failures.pop(key, None)
        self._circuit_open_until.pop(key, None)
        self._backoff_level.pop(key, None)
        self._backoff_maxed.discard(key)
        self._broken.discard(key)

    def _record_success(self, key: Any) -> None:
        self._clear_circuit_state(key)
        self._last_ok[key] = True

    def _record_failure(self, key: Any) -> None:
        self._last_ok[key] = False
        count = self._failures.get(key, 0) + 1
        self._failures[key] = count
        if count < self._cb_threshold:
            return

        if key in self._backoff_maxed:
            # Already waited out one full max-length cooldown and failed again — that's
            # enough evidence this isn't a transient blip. Stop auto-retrying.
            self._broken.add(key)
            self._circuit_open_until.pop(key, None)
            self._backoff_maxed.discard(key)
            logger.error(
                "Integration %r marked BROKEN after sustained failures at max backoff; "
                "call invalidate() to retry",
                key,
            )
            return

        level = self._backoff_level.get(key, 0)
        cooldown = min(self._cb_cooldown * (2**level), self._cb_max_cooldown)
        self._circuit_open_until[key] = time.monotonic() + cooldown
        self._backoff_level[key] = level + 1
        if cooldown >= self._cb_max_cooldown:
            self._backoff_maxed.add(key)
        logger.warning(
            "Circuit breaker open for %r after %d consecutive failures; cooling down for %.0fs",
            key,
            count,
            cooldown,
        )

    async def get(self, key: Any, fetch: Callable[[], Awaitable[Any]]) -> Any:
        """Return the cached value for ``key``, per the rules described above.

        ``fetch`` is an async, no-arg callable performing the live I/O.
        """
        now = time.monotonic()
        entry = self._entries.get(key)

        if entry is not None:
            if entry.expires_at > now:
                return entry.value
            self._maybe_schedule_refresh(key, fetch)
            return entry.value

        if self._circuit_open(key):
            return self._empty()

        lock = self._lock_for(key)
        async with lock:
            entry = self._entries.get(key)
            if entry is not None:
                return entry.value
            try:
                value = await asyncio.wait_for(fetch(), timeout=self._timeout)
            except Exception:
                logger.warning("Integration fetch failed/timed out for %r", key, exc_info=True)
                self._record_failure(key)
                self._maybe_schedule_refresh(key, fetch)
                return self._empty()
            self._record_success(key)
            self._entries[key] = _CacheEntry(value, time.monotonic() + self._ttl)
            return value

    def _maybe_schedule_refresh(self, key: Any, fetch: Callable[[], Awaitable[Any]]) -> None:
        """Fire off a background refresh for ``key``, unless one's already in flight
        or the circuit is open. Fire-and-forget: callers keep serving the stale/empty
        value immediately and don't await this.
        """
        if key in self._refreshing or self._circuit_open(key):
            return
        self._refreshing.add(key)

        async def _refresh() -> None:
            try:
                async with self._lock_for(key):
                    value = await asyncio.wait_for(fetch(), timeout=self._timeout)
                    self._record_success(key)
                    self._entries[key] = _CacheEntry(value, time.monotonic() + self._ttl)
            except Exception:
                logger.warning("Background integration refresh failed for %r", key, exc_info=True)
                self._record_failure(key)
            finally:
                self._refreshing.discard(key)

        asyncio.create_task(_refresh())  # noqa: RUF006 — not awaited

    def invalidate(self, key: Any) -> None:
        """Reset a key to a fresh state. The only way to recover a BROKEN key."""
        self._entries.pop(key, None)
        self._clear_circuit_state(key)
        self._last_ok.pop(key, None)

    async def warm(self, key: Any, fetch: Callable[[], Awaitable[Any]]) -> None:
        """Populate the cache for ``key`` if it isn't already cached.

        Fetch failures are already handled by ``get()``, which records them and
        returns ``empty()`` instead of raising.
        """
        if key in self._entries or self._circuit_open(key):
            return
        await self.get(key, fetch)
