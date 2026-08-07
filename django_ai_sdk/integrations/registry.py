"""Process-wide registry of integration services.

Integrations register themselves from their app's ready() (see IntegrationAppConfig
in apps.py). This module is just the shared dict they register into and the lookups
the agent and the host project's integrations endpoints use.

get_all_integrations() and get_integrations() are async even though this
implementation needs no I/O, to leave room for a database-backed source without
changing any caller.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from django_ai_sdk.integrations.base import Integration

logger = logging.getLogger(__name__)

_registry: dict[str, Integration] = {}

#: Whether the orphaned-config warning below has already run. Once per process: the
#: condition can't change without a restart, and this is on the read path.
_checked_orphans = False


def register(service: Integration) -> None:
    """Register (or replace) an integration service under its name."""
    if not service.name:
        raise ValueError(f"{type(service).__name__} must set a non-empty `name` to register")
    _registry[service.name] = service


def _warn_orphaned_config() -> None:
    """Name AI_SDK_INTEGRATIONS entries whose app was never installed.

    Configuring an integration and forgetting its INSTALLED_APPS entry is the easiest
    mistake to make here, and until this warning it was completely silent: the config
    was ignored and the integration simply didn't exist, which reads like a bug in the
    SDK rather than a missing line in settings.py.
    """
    global _checked_orphans
    if _checked_orphans:
        return
    _checked_orphans = True

    from django_ai_sdk.integrations.config import configured_names

    orphaned = configured_names() - set(_registry)
    if orphaned:
        logger.warning(
            "AI_SDK_INTEGRATIONS configures %s, but no installed app registers them; "
            "add each integration's app to INSTALLED_APPS",
            ", ".join(sorted(repr(name) for name in orphaned)),
        )


async def get_all_integrations() -> dict[str, Integration]:
    """Return every registered integration service, keyed by name."""
    _warn_orphaned_config()
    return dict(_registry)


async def get_integrations(names: list[str]) -> dict[str, Integration]:
    """Return the services named in names. Unknown names are skipped."""
    _warn_orphaned_config()
    return {name: _registry[name] for name in names if name in _registry}


def reset_registry() -> None:
    """Clear the registry — for tests."""
    global _checked_orphans
    _registry.clear()
    _checked_orphans = False
