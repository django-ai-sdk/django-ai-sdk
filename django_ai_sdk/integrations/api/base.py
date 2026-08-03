"""Base class for hand-written API-backed integrations."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from django.conf import settings

from django_ai_sdk.integrations.base import (
    Integration,
    IntegrationStatus,
    ResilientCache,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser
    from haystack.tools import Tool

    from django_ai_sdk.assistant import Assistant


def _accepted_kwargs(factory: Callable[..., Any], context: dict[str, Any]) -> dict[str, Any]:
    """Filter context to the keys factory actually accepts.

    Lets a tool factory declare only the request context it needs: def f(user), def
    f(user, thread_id), and def f(**kwargs) all work, instead of being forced to
    accept every argument.
    """
    params = inspect.signature(factory).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return context
    return {k: v for k, v in context.items() if k in params}


class APIIntegration(Integration):
    """Base for integrations backed by a hand-written API client, not MCP.

    Each entry in tools is either:

    - a ready-made Haystack Tool, the simple, common case. Decorate a plain typed
      function with @haystack.tools.tool and the schema is inferred from its type
      hints and docstring; no factory, no hand-written parameter schema:

          @tool
          def get_current_weather(location: str) -> dict:
              \"\"\"Get the current weather for a place name.\"\"\"
              ...

          class WeatherIntegration(APIIntegration):
              name = "weather"
              tools = [get_current_weather]

      Plain type hints work fine under from __future__ import annotations. Only
      per-parameter descriptions via Annotated[str, "..."] need the hints to be real
      objects; for those, either pass an explicit parameters= schema or define the
      tool in a module without the __future__ import.

    - a factory callable returning a Tool or list of Tools (optionally async), for
      tools that must be built per request, e.g. one carrying the current user's API
      token. A factory is passed only the arguments it declares: any subset of
      user, assistant, thread_id (or all of them via **kwargs). So def
      make_tool(user): ... and def make_tool(**kwargs): ... both work:

          def issue_tool(user):
              return build_authed_tool(token=user.tracker_token)

          class IssueTrackerIntegration(APIIntegration):
              name = "issue_tracker"
              tools = [issue_tool]

    get_status() reports ACTIVE unconditionally unless a subclass sets
    health_check, an async, no-arg callable that raises on failure. Assign it as a
    staticmethod (health_check = staticmethod(check_api)) or as an async def
    health_check(self) method; a bare function assigned to the attribute would be
    bound and wrongly receive self. When set, it's run through the same
    ResilientCache (stale-while-revalidate plus circuit breaker; see
    django_ai_sdk.integrations.base) every other integration kind uses, so a down
    API shows up as DEGRADED instead of a false ACTIVE.
    """

    name: str = ""
    label: str = ""
    tools: list[Tool | Callable[..., Any]] = []
    health_check: Callable[[], Awaitable[Any]] | None = None

    def __init__(self) -> None:
        # Derive a label from `name` only. Falling back to the class name would produce
        # things like "Unnamedservice", and would also pre-empt the registry's better
        # fallback (the settings key) for a service that declares no name.
        if not self.label and self.name:
            self.label = self.name.title()
        # Decided once, from the class-level health_check — set it as a class
        # attribute (as in the docstring example), not on an instance after __init__.
        self._cache: ResilientCache | None = (
            ResilientCache(
                ttl=getattr(settings, "AI_SDK_INTEGRATION_CACHE_TTL", 900),
                timeout=getattr(settings, "AI_SDK_INTEGRATION_TIMEOUT", 3),
                cb_cooldown=getattr(settings, "AI_SDK_INTEGRATION_CB_COOLDOWN", 60),
            )
            if self.health_check is not None
            else None
        )

    async def get_tools(
        self,
        user: AbstractBaseUser | AnonymousUser | None = None,
        assistant: Assistant | None = None,
        thread_id: str = "",
    ) -> list[Any]:
        from haystack.tools import Tool

        context = {"user": user, "assistant": assistant, "thread_id": thread_id}
        result: list[Any] = []
        for entry in self.tools:
            # A ready-made Tool (e.g. a @tool-decorated function) is used as-is;
            # anything else is a factory, called with only the context args it declares.
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
        assistant: Assistant | None = None,
    ) -> IntegrationStatus:
        if self._cache is None or self.health_check is None:
            return IntegrationStatus.ACTIVE
        await self._cache.get(self.name, self.health_check)
        return self._cache.status_for(self.name)

    async def reconnect(self, user: AbstractBaseUser | AnonymousUser | None = None) -> None:
        """Reset a degraded health check to a fresh state and retry on the next check."""
        if self._cache is not None:
            await self._cache.invalidate(self.name)
