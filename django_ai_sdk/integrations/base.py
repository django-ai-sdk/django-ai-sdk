from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from cashews import Cache, CircuitBreakerOpen

from django_ai_sdk.errors import UserError
from django_ai_sdk.permissions import Operation, PermissionDomain

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.agents.base import Agent
    from django_ai_sdk.permissions import BasePermission

logger = logging.getLogger(__name__)


class IntegrationNotConnectable(UserError):
    """Raised when connect() is called on an integration that doesn't support it."""


class IntegrationStatus(StrEnum):
    """ACTIVE/DEGRADED come from the circuit breaker; EXPIRED/DISCONNECTED from credentials."""

    ACTIVE = "active"
    DEGRADED = "degraded"
    EXPIRED = "expired"
    DISCONNECTED = "disconnected"


class Integration(ABC):
    """One integration: its tools, health, permissions and connection lifecycle."""

    name: str = ""
    label: str = ""

    #: Prepended to every tool's description, e.g. "Company wiki and HR docs".
    hint: str = ""

    permissions: list[type[BasePermission] | BasePermission] = []
    domain: PermissionDomain = PermissionDomain.INTEGRATIONS

    supports_connect: bool = False
    supports_test: bool = True

    #: "oauth" or None; a string so more flows fit without a contract change.
    connect_kind: str | None = None

    #: Why this integration can't run yet (e.g. a missing secret), or None.
    detail: str | None = None

    def secret(self, key: str, default: str = "") -> str:
        """AI_SDK_INTEGRATIONS[self.name][KEY] as a string, or ``default`` when unset."""
        value = self._config().get(key.upper())
        if value is None:
            return default
        if not isinstance(value, str):
            # Never stringify or log the value: it is a secret.
            logger.warning(
                "AI_SDK_INTEGRATIONS[%r][%r] must be a string, got %s; treating it as unset",
                self.name,
                key.upper(),
                type(value).__name__,
            )
            return default
        return value or default

    def _config(self) -> dict[str, Any]:
        from django_ai_sdk.integrations.config import get_integration_config

        return get_integration_config(self.name)

    @abstractmethod
    async def get_tools(
        self,
        user: AbstractBaseUser | AnonymousUser | None = None,
        agent: Agent | None = None,
        thread_id: str = "",
    ) -> list[Any]:
        """Return this integration's tool objects."""

    @abstractmethod
    async def get_status(
        self,
        user: AbstractBaseUser | AnonymousUser | None = None,
        agent: Agent | None = None,
    ) -> IntegrationStatus:
        """Return this integration's current health."""

    @property
    def kind(self) -> str:
        return "api"

    async def get_tool_names(
        self, user: AbstractBaseUser | AnonymousUser | None = None
    ) -> list[str]:
        """Tool names for display; the default calls get_tools(), which may do I/O."""
        return [t.name for t in await self.get_tools(user)]

    async def reconnect(self, user: AbstractBaseUser | AnonymousUser | None = None) -> None:
        """Reset this integration to a fresh state. No-op by default."""

    async def connect(
        self,
        user: AbstractBaseUser | AnonymousUser | None = None,
        *,
        request: Any = None,
        redirect_uri: str = "",
    ) -> dict[str, Any]:
        """Begin connecting for user; returns e.g. {"redirect_url": ...}.

        Raises IntegrationNotConnectable unless supports_connect.
        """
        raise IntegrationNotConnectable(f"{self.name!r} does not support connect()")

    async def disconnect(self, user: AbstractBaseUser | AnonymousUser | None = None) -> bool:
        """Drop any stored credential/connection for user. No-op default."""
        return False

    async def test(self, user: AbstractBaseUser | AnonymousUser | None = None) -> IntegrationStatus:
        """Force a fresh connection attempt and report the real outcome."""
        await self.reconnect(user)
        return await self.get_status(user)

    async def refresh(self, user: AbstractBaseUser | AnonymousUser | None = None) -> None:
        """Refresh credentials (e.g. rotate an OAuth token). No-op by default."""

    async def has_perms(
        self,
        user: AbstractBaseUser | AnonymousUser | None,
        operation: Operation = Operation.USE_INTEGRATION,
        *,
        raise_on_deny: bool = False,
    ) -> bool:
        """Whether user may use/manage this integration."""
        from django_ai_sdk.permissions import get_integration_permissions
        from django_ai_sdk.permissions import has_perms as _has_perms

        return await _has_perms(
            user,
            operation,
            permissions=get_integration_permissions(self),
            raise_on_deny=raise_on_deny,
        )


# Deliberately not settings, to avoid per-integration tuning.
_CB_ERRORS_RATE = 50
_CB_MIN_CALLS = 3
_CB_PERIOD = 600


class ResilientCache:
    """Stale-while-revalidate cache with a per-key circuit breaker; each instance is isolated."""

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
        # cashews exposes no breaker state, so track it for status_for().
        self._last_ok: dict[str, bool] = {}
        # _last_ok can't tell "just failed" from "breaker open"; this can.
        self._breaker_open: set[str] = set()
        self._cache = Cache()
        self._cache.setup("mem://")
        # A float, so it stays below ttl for sub-second TTLs too.
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
            return await asyncio.wait_for(fetch(), timeout=self._timeout)

        self._run = _run

    @staticmethod
    def _norm(key: Any) -> str:
        return str(key)

    async def get(self, key: Any, fetch: Callable[[], Awaitable[Any]]) -> Any:
        """Return the cached value for key; never raises, returns empty() on failure."""
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
        except Exception as exc:
            # Expected while a server is down, so the traceback is debug only.
            logger.warning("Integration fetch failed/timed out for %r: %r", key, exc)
            logger.debug("Integration fetch failure for %r", key, exc_info=True)
            self._last_ok[k] = False
            self._breaker_open.discard(k)
            return self._empty()
        self._last_ok[k] = True
        if was_ok is False or k in self._breaker_open:
            logger.info("Integration %r: recovered — fetch succeeded again.", key)
        self._breaker_open.discard(k)
        return value

    def status_for(self, key: Any) -> IntegrationStatus:
        """Status from the last attempt for key; raises if get() never ran for it."""
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
        """Drop cached values and reset the breaker, forcing a retry now.

        Clears the whole backend (one integration's keys) to avoid cashews' key layout.
        """
        await self._cache.clear()
        self._last_ok.pop(self._norm(key), None)
        self._breaker_open.discard(self._norm(key))
