from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from django_ai_sdk.integrations.base import (
    Integration,
    IntegrationStatus,
    ResilientCache,
)
from django_ai_sdk.utils import resolve_setting

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser
    from haystack.tools import Tool

    from django_ai_sdk.agents.base import Agent


def _accepted_kwargs(factory: Callable[..., Any], context: dict[str, Any]) -> dict[str, Any]:
    """Filter context to the keys factory accepts."""
    params = inspect.signature(factory).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return context
    return {k: v for k, v in context.items() if k in params}


class APIIntegration(Integration):
    """Base for integrations backed by a hand-written API client, not MCP.

    `tools`: Haystack Tools, or factories passed whichever of user, agent, thread_id they declare.
    `health_check`: async no-arg staticmethod that raises on failure; without it, always ACTIVE.
    """

    name: str = ""
    label: str = ""
    tools: list[Tool | Callable[..., Any]] = []
    health_check: Callable[[], Awaitable[Any]] | None = None

    def __init__(self) -> None:
        if not self.label and self.name:
            self.label = self.name.title()
        # Decided once: set health_check on the class, not the instance.
        self._cache: ResilientCache | None = (
            ResilientCache(
                ttl=resolve_setting("AI_SDK_INTEGRATION_CACHE_TTL", 900),
                timeout=resolve_setting("AI_SDK_INTEGRATION_TIMEOUT", 3),
                cb_cooldown=resolve_setting("AI_SDK_INTEGRATION_CB_COOLDOWN", 60),
            )
            if self.health_check is not None
            else None
        )

    async def get_tools(
        self,
        user: AbstractBaseUser | AnonymousUser | None = None,
        agent: Agent | None = None,
        thread_id: str = "",
    ) -> list[Any]:
        from haystack.tools import Tool

        context = {"user": user, "agent": agent, "thread_id": thread_id}
        result: list[Any] = []
        for entry in self.tools:
            if isinstance(entry, Tool):
                result.append(entry)
                continue
            items = entry(**_accepted_kwargs(entry, context))
            if inspect.isawaitable(items):
                items = await items
            if isinstance(items, list):
                result.extend(items)
            else:
                result.append(items)
        return result

    async def get_status(
        self,
        user: AbstractBaseUser | AnonymousUser | None = None,
        agent: Agent | None = None,
    ) -> IntegrationStatus:
        if self._cache is None or self.health_check is None:
            return IntegrationStatus.ACTIVE
        await self._cache.get(self.name, self.health_check)
        return self._cache.status_for(self.name)

    async def reconnect(self, user: AbstractBaseUser | AnonymousUser | None = None) -> None:
        """Reset a degraded health check to a fresh state and retry on the next check."""
        if self._cache is not None:
            await self._cache.invalidate(self.name)
