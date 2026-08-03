"""Process-wide registry of integration services.

Integrations register themselves from their app's ready() (see IntegrationAppConfig
in apps.py). This module is just the shared dict they register into and the lookups
the assistant and the /api/integrations router use.

get_all_integrations() and get_integrations() are async even though this
implementation needs no I/O, to leave room for a database-backed source without
changing any caller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from django_ai_sdk.integrations.base import Integration

_registry: dict[str, Integration] = {}


def register(service: Integration) -> None:
    """Register (or replace) an integration service under its name."""
    if not service.name:
        raise ValueError(f"{type(service).__name__} must set a non-empty `name` to register")
    _registry[service.name] = service


async def get_all_integrations() -> dict[str, Integration]:
    """Return every registered integration service, keyed by name."""
    return dict(_registry)


async def get_integrations(names: list[str]) -> dict[str, Integration]:
    """Return the services named in names. Unknown names are skipped."""
    return {name: _registry[name] for name in names if name in _registry}


def reset_registry() -> None:
    """Clear the registry — for tests."""
    _registry.clear()
