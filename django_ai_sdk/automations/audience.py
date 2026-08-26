"""Who an automation runs as.

Two are shipped; implement the `AudienceResolver` protocol for anything else.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AudienceResolver(Protocol):
    """Resolves the principals one tick should produce runs for."""

    async def resolve(self, automation: Any) -> list[Any]:
        """The principals for this tick. Empty means nothing to do."""
        ...

    def describe(self) -> str:
        """Short human-readable form, for the API."""
        ...


class _AppPrincipal:
    async def resolve(self, automation: Any) -> list[Any]:
        return [None]

    def describe(self) -> str:
        return "app"

    def __repr__(self) -> str:
        return "Audience.APP"


class _SubscribedUsers:
    async def resolve(self, automation: Any) -> list[Any]:
        from django.contrib.auth import get_user_model

        return [
            user
            async for user in get_user_model().objects.filter(
                automation_subscriptions__name=automation.name,
                automation_subscriptions__enabled=True,
            )
        ]

    def describe(self) -> str:
        return "subscribed"

    def __repr__(self) -> str:
        return "Audience.SUBSCRIBED"


class Audience:
    """Namespace of the two built-in resolvers. Never instantiated."""

    # One run, no user: the deployment acting as itself.
    APP = _AppPrincipal()

    # One run per user who opted in.
    SUBSCRIBED = _SubscribedUsers()


__all__ = ["Audience", "AudienceResolver"]
