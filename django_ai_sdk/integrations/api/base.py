"""Base class for hand-written API-backed integrations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django.conf import settings

from django_ai_sdk.integrations.base import Integration, IntegrationStatus, ResilientCache

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.assistant import Assistant


class APIIntegration(Integration):
    """Base for integrations backed by a hand-written API client, not MCP.

    Each entry in ``tools`` is a factory called as ``factory(user=..., assistant=...)``,
    returning a Tool or a list of Tools.

    ``get_status()`` reports ACTIVE unconditionally unless a subclass sets
    ``health_check`` — an async, no-arg callable that raises on failure. When set, it's
    run through the same ResilientCache (stale-while-revalidate + circuit breaker; see
    ``django_ai_sdk.integrations.base``) every other integration kind uses, so a down
    API shows up as DEGRADED/BROKEN instead of a false ACTIVE.

    A subclass only needs to set class attributes — see
    ``piratespeak.integrations.weather.WeatherIntegration`` in the demo app for a
    runnable example::

        class WeatherIntegration(APIIntegration):
            label = "Weather"
            tools = [get_weather_tool]
            health_check = staticmethod(check_weather_api)
    """

    name: str = ""
    label: str = ""
    tools: list[Callable[..., Any]] = []
    health_check: Callable[[], Awaitable[Any]] | None = None

    def __init__(self) -> None:
        if not self.label:
            self.label = (self.name or type(self).__name__).title()
        # Decided once, from the class-level health_check — set it as a class
        # attribute (as in the docstring example), not on an instance after __init__.
        self._cache: ResilientCache | None = (
            ResilientCache(
                ttl=getattr(settings, "AI_SDK_INTEGRATION_CACHE_TTL", 300),
                timeout=getattr(settings, "AI_SDK_INTEGRATION_TIMEOUT", 5),
                cb_threshold=getattr(settings, "AI_SDK_INTEGRATION_CB_THRESHOLD", 3),
                cb_cooldown=getattr(settings, "AI_SDK_INTEGRATION_CB_COOLDOWN", 60),
                cb_max_cooldown=getattr(settings, "AI_SDK_INTEGRATION_CB_MAX_COOLDOWN", 1800),
            )
            if self.health_check is not None
            else None
        )

    async def get_tools(
        self,
        user: AbstractBaseUser | AnonymousUser | None = None,
        assistant: Assistant | None = None,
    ) -> list[Any]:
        result: list[Any] = []
        for factory in self.tools:
            items = factory(user=user, assistant=assistant)
            if isinstance(items, list):
                result.extend(items)
            else:
                result.append(items)
        return result

    async def get_status(
        self,
        user: AbstractBaseUser | AnonymousUser | None = None,
        assistant: Assistant | None = None,
    ) -> IntegrationStatus:
        if self._cache is None or self.health_check is None:
            return IntegrationStatus.ACTIVE
        await self._cache.get(self.name, self.health_check)
        return self._cache.status_for(self.name)

    async def reconnect(self, user: AbstractBaseUser | AnonymousUser | None = None) -> None:
        """Reset a degraded/broken health check back to a fresh, never-attempted state."""
        if self._cache is not None:
            self._cache.invalidate(self.name)
