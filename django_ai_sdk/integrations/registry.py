"""Process-wide registry of integration services.

Integrations are declared in code, as a mapping of registry name to the dotted path
of an :class:`~django_ai_sdk.integrations.base.IntegrationService` subclass::

    AI_SDK_INTEGRATIONS = {
        "notion": "django_ai_sdk.integrations.defaults.NotionService",
        "linear": "django_ai_sdk.integrations.defaults.LinearService",
        "weather": "myapp.integrations.WeatherService",
    }

A value may also be a ready-made ``IntegrationService`` instance, for a service that
needs constructor arguments (or in tests).

Each is instantiated once, lazily, and cached for the life of the process — so each
service keeps its ``ResilientCache`` and circuit-breaker state across requests instead
of losing it on every lookup.

An integration that needs its own Django app (models, admin, migrations) can instead
subclass :class:`~django_ai_sdk.integrations.apps.IntegrationAppConfig` and register
itself from ``ready()``; both paths write to the same registry. The settings mapping is
the common case.

``get_all_integrations``/``get_integrations`` are ``async`` even though this
implementation needs no I/O. That is deliberate: it keeps the door open for a
second, database-backed source of integrations (admin-managed MCP servers, which do
need a query) without changing a single caller. A synchronous caller can bridge with
``asgiref.sync.async_to_sync``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string

if TYPE_CHECKING:
    from django_ai_sdk.integrations.base import IntegrationService

logger = logging.getLogger(__name__)

#: Explicitly registered services (see `register`), and instances built from
#: AI_SDK_INTEGRATIONS. Keyed by registry name.
_registry: dict[str, IntegrationService] = {}

#: Names from AI_SDK_INTEGRATIONS that failed to build. Remembered so a broken entry
#: is reported once rather than on every request, and never retried into a hot path.
_broken: set[str] = set()


def register(service: IntegrationService) -> None:
    """Register (or replace) an integration service under its ``name``.

    Used by ``IntegrationAppConfig.ready()`` and by tests. Services declared in
    ``AI_SDK_INTEGRATIONS`` don't need this — they're built on first access.
    """
    if not service.name:
        raise ValueError(f"{type(service).__name__} must set a non-empty `name` to register")
    _registry[service.name] = service


def _configured() -> dict[str, str | IntegrationService]:
    """The ``AI_SDK_INTEGRATIONS`` mapping of name → dotted path (or instance)."""
    configured = getattr(settings, "AI_SDK_INTEGRATIONS", {}) or {}
    if not isinstance(configured, dict):
        raise ImproperlyConfigured(
            "AI_SDK_INTEGRATIONS must be a dict of {name: 'dotted.path.To.Service'}, "
            f"got {type(configured).__name__}"
        )
    return configured


def _build(name: str, entry: str | IntegrationService) -> IntegrationService | None:
    """Resolve one configured entry to a service. Returns None (and logs) on failure.

    ``entry`` is normally a dotted path, instantiated with no arguments. A ready-made
    ``IntegrationService`` instance is also accepted, for the cases a dotted path can't
    express — a service needing constructor arguments, or a test fixture.

    A misconfigured integration must not take down every other integration — nor the
    chat request that happened to trigger the first lookup — so this degrades to
    "that one is missing" rather than raising.
    """
    try:
        service = import_string(entry)() if isinstance(entry, str) else entry
    except Exception:
        logger.exception(
            "Could not load integration %r from %r — it will be unavailable", name, entry
        )
        _broken.add(name)
        return None

    # The settings key is authoritative: it's what assistants and URLs reference, so a
    # service that omits `name` (or disagrees with its key) is aligned to the key here.
    if service.name != name:
        service.name = name
    if not service.label:
        service.label = name.title()
    _registry[name] = service
    return service


async def get_all_integrations() -> dict[str, IntegrationService]:
    """Return every available integration service, keyed by name."""
    result: dict[str, IntegrationService] = {}
    for name, path in _configured().items():
        service = _registry.get(name)
        if service is None:
            if name in _broken:
                continue
            service = _build(name, path)
            if service is None:
                continue
        result[name] = service
    # Explicitly registered services (app-based, or test fixtures) win on a name
    # collision — they were constructed deliberately rather than resolved from config.
    return {**result, **_registry}


async def get_integrations(names: list[str]) -> dict[str, IntegrationService]:
    """Return the services named in ``names``. Unknown names are skipped."""
    all_integrations = await get_all_integrations()
    return {name: all_integrations[name] for name in names if name in all_integrations}


def reset_registry() -> None:
    """Clear the registry — for tests that register their own services, or that change
    ``AI_SDK_INTEGRATIONS`` and need the next lookup to rebuild from it."""
    _registry.clear()
    _broken.clear()
