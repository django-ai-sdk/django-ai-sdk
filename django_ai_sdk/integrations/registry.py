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

_checked_orphans = False

_warned_shadowed: set[str] = set()

#: Live objects per DB row, kept until updated_at changes, so breaker state survives.
_db_cache: dict[str, tuple[datetime, Integration]] = {}

#: In Django's cache so a shared backend invalidates across workers.
_ROWS_CACHE_KEY = "django_ai_sdk:mcp_server_config:enabled_rows"


def invalidate_db_rows_cache() -> None:
    """Drop the cached enabled-rows list."""
    cache.delete(_ROWS_CACHE_KEY)


def register(service: Integration) -> None:
    """Register (or replace) an integration; warns when a different class has the name."""
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
    """Warn once about AI_SDK_INTEGRATIONS entries whose app was never installed."""
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
    """Enabled MCPServerConfig rows; an installed app wins a name clash. Never raises."""
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
