"""System checks for the receive side: the dedup cache, and pointing an agent at it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django.conf import settings
from django.core.checks import Warning as CheckWarning

if TYPE_CHECKING:
    from django.core.checks import CheckMessage


def _webhook_integration_names() -> set[str]:
    """Names of the receive-only integrations installed apps declare.

    Code-declared only, and by class rather than by instance: the checks run before
    anything has resolved configuration, and must not touch the database.
    """
    from django.apps import apps as django_apps
    from django.utils.module_loading import import_string

    from django_ai_sdk.integrations.webhooks.base import WebhookIntegration

    names: set[str] = set()
    for config in django_apps.get_app_configs():
        path = getattr(config, "integration", "")
        if not path:
            continue
        try:
            declared = import_string(path)
        except ImportError:
            continue
        if isinstance(declared, type) and issubclass(declared, WebhookIntegration):
            name = getattr(declared, "name", "")
            if name:
                names.add(name)
    return names


def check_dedup_cache(app_configs: Any, **kwargs: Any) -> list[CheckMessage]:
    """Report a default cache that cannot de-duplicate an event across processes."""
    from django.core.cache import caches
    from django.core.cache.backends.dummy import DummyCache
    from django.core.cache.backends.locmem import LocMemCache

    # runserver is one process, so a per-process cache de-duplicates correctly there
    # and the warning would only be noise.
    if settings.DEBUG or not _webhook_integration_names():
        return []

    backend = caches["default"]
    if not isinstance(backend, (LocMemCache, DummyCache)):
        return []
    return [
        CheckWarning(
            f"CACHES['default'] is {type(backend).__name__}, which no other process "
            f"can read, so a redelivered webhook event is answered once per process "
            f"instead of once.",
            hint=(
                "Point CACHES['default'] at a backend every process shares, such as "
                "Redis, Memcached or the database."
            ),
            id="ai_sdk.webhooks.W001",
        )
    ]


def check_agent_integrations(app_configs: Any, **kwargs: Any) -> list[CheckMessage]:
    """Report an agent that lists a receive-only integration among its tools.

    A webhook integration points inward: the platform pushes an event and the agent
    answers it. It has no tools to give, so naming it here resolves and contributes
    nothing — the agent is simply missing whatever the author thought they added.
    The wiring runs the other way, through AI_SDK_INTEGRATIONS[name]["AGENT"].
    """
    from django_ai_sdk.agents.registry import registry

    receive_only = _webhook_integration_names()
    if not receive_only:
        return []

    issues: list[CheckMessage] = []
    for agent_id, agent in sorted(registry.all().items()):
        named = sorted(receive_only.intersection(getattr(agent, "integrations", None) or ()))
        if not named:
            continue
        issues.append(
            CheckWarning(
                f"Agent {type(agent).__name__} ({agent_id}) lists "
                f"{', '.join(repr(n) for n in named)} in `integrations`, but "
                f"{'they are' if len(named) > 1 else 'it is'} receive-only and "
                f"contribute{'' if len(named) > 1 else 's'} no tools.",
                hint=(
                    "A webhook integration is wired the other way round: set "
                    'AI_SDK_INTEGRATIONS[name]["AGENT"] to this agent\'s dotted path '
                    "and drop the name from `integrations`."
                ),
                id="ai_sdk.webhooks.W002",
            )
        )
    return issues


__all__ = ["check_agent_integrations", "check_dedup_cache"]
