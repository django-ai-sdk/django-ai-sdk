"""Process-wide registry of integration services.

Integrations register themselves from their app's ready() (see IntegrationAppConfig
in apps.py). This module is just the shared dict they register into and the lookups
the agent and the host project's integrations endpoints use.

get_all_integrations() and get_integrations() are async even though the code-declared
side needs no I/O: enabled MCPServerConfig rows (see mcp.models.MCPServerConfig) are
merged in on every call, database-backed integrations declared by an admin/settings UI
instead of an app.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from django.core.cache import cache

from django_ai_sdk.utils import resolve_setting

if TYPE_CHECKING:
    from datetime import datetime

    from django_ai_sdk.integrations.base import Integration

logger = logging.getLogger(__name__)

_registry: dict[str, Integration] = {}

#: Whether the orphaned-config warning below has already run. Once per process: the
#: condition can't change without a restart, and this is on the read path.
_checked_orphans = False

#: Names already warned about an enabled MCPServerConfig row being shadowed by an
#: installed app (see _db_integrations) -- once per process, same as _checked_orphans.
_warned_shadowed: set[str] = set()

#: Built DynamicMCPIntegration per DB row, keyed by name, kept until updated_at
#: changes. A DB-declared server has no boot hook, so without this it would rebuild
#: (and lose its ResilientCache/circuit-breaker state) on every call. Plain
#: process-local dict, not Django's cache: it holds live objects that can't be
#: serialized, unlike the rows list below.
_db_cache: dict[str, tuple[datetime, Integration]] = {}

#: Cache key for the enabled MCPServerConfig rows list. Via Django's cache rather
#: than a bespoke store so it can be TTL-bounded and, with a shared backend
#: (Redis/Memcached), invalidated across worker processes too -- mcp.apps deletes
#: this key on save/delete for that.
_ROWS_CACHE_KEY = "django_ai_sdk:mcp_server_config:enabled_rows"


def invalidate_db_rows_cache() -> None:
    """Drop the cached enabled-rows list. Called from mcp.apps' post_save/
    post_delete signal handlers so a change is visible without waiting for the TTL.
    """
    cache.delete(_ROWS_CACHE_KEY)


def register(service: Integration) -> None:
    """Register (or replace) an integration service under its name.

    Warns when a *different* class already claims this name -- two installed apps
    declaring the same integration, where the loser would otherwise just disappear
    with no explanation. Re-registering the same class (ready() running twice) isn't
    a collision and doesn't warn.
    """
    if not service.name:
        raise ValueError(f"{type(service).__name__} must set a non-empty `name` to register")
    existing = _registry.get(service.name)
    if existing is not None and type(existing) is not type(service):
        logger.warning(
            "Integration %r is registered by both %s and %s -- only %s is reachable "
            "under that name, and which one depends on Django's app-loading order. "
            "Give one of them a different `name`.",
            service.name,
            type(existing).__name__,
            type(service).__name__,
            type(service).__name__,
        )
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


async def _db_integrations() -> dict[str, Integration]:
    """Enabled MCPServerConfig rows not already provided by an installed app.

    An installed app always wins on a name collision -- a DB row is for adding a
    server with no code at all, not overriding one. Degrades to {} (never raises)
    when the MCP toolkit app isn't installed or its table doesn't exist yet.
    """
    from django.apps import apps as django_apps

    if not django_apps.is_installed("django_ai_sdk.integrations.mcp"):
        return {}

    from django_ai_sdk.integrations.mcp.models import MCPServerConfig

    ttl = resolve_setting("AI_SDK_MCP_SERVER_LIST_CACHE_TTL", 30)
    rows = await cache.aget(_ROWS_CACHE_KEY)
    if rows is None:
        try:
            rows = [row async for row in MCPServerConfig.objects.filter(enabled=True)]
        except Exception:
            logger.exception("Failed to load DB-declared MCP servers; skipping for this call")
            return {}
        await cache.aset(_ROWS_CACHE_KEY, rows, timeout=ttl)

    result: dict[str, Integration] = {}
    for row in rows:
        if row.name in _registry:
            if row.enabled and row.name not in _warned_shadowed:
                _warned_shadowed.add(row.name)
                logger.warning(
                    "MCPServerConfig %r is enabled, but an installed app already "
                    "registers an integration under that name -- the installed app "
                    "always wins, so this row is never used. Rename the row (or "
                    "resolve the collision on the code side) if that's not intended.",
                    row.name,
                )
            continue
        cached = _db_cache.get(row.name)
        if cached is not None and cached[0] == row.updated_at:
            result[row.name] = cached[1]
            continue
        integration = row.to_integration()
        _db_cache[row.name] = (row.updated_at, integration)
        result[row.name] = integration
    return result


async def get_all_integrations() -> dict[str, Integration]:
    """Return every registered integration service, keyed by name."""
    _warn_orphaned_config()
    merged = dict(_registry)
    merged.update(await _db_integrations())
    return merged


async def get_integrations(names: list[str]) -> dict[str, Integration]:
    """Return the services named in names. Unknown names are skipped."""
    _warn_orphaned_config()
    merged = dict(_registry)
    if set(names) - set(merged):
        merged.update(await _db_integrations())
    return {name: merged[name] for name in names if name in merged}


def reset_registry() -> None:
    """Clear the registry — for tests."""
    global _checked_orphans
    _registry.clear()
    _db_cache.clear()
    _warned_shadowed.clear()
    invalidate_db_rows_cache()
    _checked_orphans = False
