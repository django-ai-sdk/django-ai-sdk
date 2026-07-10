"""Process-wide registry of integration services.

Two sources merge, keyed by name:

- **App-registered** (code): each integration is a Django app whose
  ``IntegrationAppConfig.ready()`` constructs its ``IntegrationService`` and calls
  :func:`register`. Being listed in ``INSTALLED_APPS`` is what enables it.
- **DB-declared** (data): ``MCPServerConfig`` rows, for MCP servers that are pure
  config (URL + auth + tool allow-list) and don't warrant a code app — added, edited,
  enabled/disabled from Django admin with no deploy. See :func:`_db_services`.

A name registered both ways is served from the app (code wins); a warning is logged.

``get_all_integrations``/``get_integrations`` are async because the DB-declared source
does a real query — matching this codebase's convention of async ORM access
(``aget``/``async for``) everywhere a lookup runs inside a request/async call chain.
The one legitimate synchronous caller (the ``manage.py check`` system check) bridges
with ``asgiref.sync.async_to_sync``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from django_ai_sdk.integrations.base import IntegrationService

logger = logging.getLogger(__name__)

_registry: dict[str, IntegrationService] = {}

# DB-declared services are cached per name so each one keeps its ResilientCache/circuit
# breaker across requests instead of losing that state on every rebuild. Invalidated by
# post_save/post_delete signals on MCPServerConfig (see integrations/mcp/apps.py).
_db_cache: dict[str, IntegrationService] = {}


def register(service: IntegrationService) -> None:
    """Register (or replace) an integration service under its ``name``."""
    if not service.name:
        raise ValueError(f"{type(service).__name__} must set a non-empty `name` to register")
    _registry[service.name] = service


async def _db_services() -> dict[str, IntegrationService]:
    """Build (or return cached) services for every enabled ``MCPServerConfig`` row.

    Returns ``{}`` if the mcp toolkit's tables aren't migrated yet (e.g. ``manage.py
    check`` on a fresh checkout, before the first ``migrate``) — a missing table must
    never crash a call to :func:`get_all_integrations`.
    """
    from django.db import DatabaseError

    from django_ai_sdk.integrations.mcp.loader import MCPIntegration
    from django_ai_sdk.integrations.mcp.models import MCPServerConfig

    result: dict[str, IntegrationService] = {}
    try:
        async for row in MCPServerConfig.objects.filter(enabled=True):
            svc = _db_cache.get(row.name)
            if svc is None:
                config, needs_setup = row.to_config()
                svc = MCPIntegration(row.name, config, needs_setup=needs_setup)
                _db_cache[row.name] = svc
            result[row.name] = svc
    except DatabaseError:
        return {}
    return result


async def get_all_integrations() -> dict[str, IntegrationService]:
    """Return every integration service, keyed by name — app-registered + DB-declared.

    App-registered wins on a name collision.
    """
    db = await _db_services()
    collisions = db.keys() & _registry.keys()
    for name in collisions:
        logger.warning(
            "Integration %r is both an app-registered service and a DB MCPServerConfig "
            "row; the app-registered service wins.",
            name,
        )
    return {**db, **_registry}


async def get_integrations(names: list[str]) -> dict[str, IntegrationService]:
    """Return the services named in ``names``. Unknown names are skipped."""
    all_integrations = await get_all_integrations()
    return {name: all_integrations[name] for name in names if name in all_integrations}


def invalidate_db_service(name: str) -> None:
    """Drop a DB-declared service's cached instance so the next access rebuilds it
    from the current row — picks up an edited config or an enabled/disabled toggle.
    Connected to ``MCPServerConfig`` post_save/post_delete (see ``mcp/apps.py``).
    """
    _db_cache.pop(name, None)


def reset_registry() -> None:
    """Clear both registries — for tests that register their own services."""
    _registry.clear()
    _db_cache.clear()
