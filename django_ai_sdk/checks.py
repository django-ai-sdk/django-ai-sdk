"""System checks for the core app. Ids are `ai_sdk.E001` and up."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django.core.checks import Error, Warning  # noqa: A004

if TYPE_CHECKING:
    from collections.abc import Callable

PREFIX = "ai_sdk"
OVERSIGHT_APP = "django_ai_sdk.oversight"


def check_oversight_installed(app_configs: Any = None, **kwargs: Any) -> list[Any]:
    """Report policies that are set while the oversight app is not installed.

    Those agents run unguarded: `oversight`, a `get_oversight()` override or DEFAULT_POLICY.
    """
    from django.apps import apps
    from django.conf import settings

    if apps.is_installed(OVERSIGHT_APP):
        return []
    messages: list[Any] = []
    hint = f"Add {OVERSIGHT_APP!r} to INSTALLED_APPS, or remove the policy."
    configured = getattr(settings, "AI_SDK_OVERSIGHT", None)
    default = configured.get("DEFAULT_POLICY") if isinstance(configured, dict) else None
    if default:
        messages.append(
            Error(
                f"AI_SDK_OVERSIGHT['DEFAULT_POLICY'] is {default!r}, but {OVERSIGHT_APP} is "
                "not installed: no turn is checked.",
                hint=hint,
                id=f"{PREFIX}.E001",
            )
        )
    from django_ai_sdk.agents.registry import registered_agent_classes

    for cls in registered_agent_classes():
        if _asks_for_policy(cls):
            messages.append(
                Error(
                    f"Agent {cls.__qualname__} asks for an oversight policy, but "
                    f"{OVERSIGHT_APP} is not installed: its turns run unguarded.",
                    hint=hint,
                    id=f"{PREFIX}.E001",
                )
            )
    return messages


# Caches that each process keeps to itself, or that keep nothing.
_UNSHARED_CACHES = (
    "django.core.cache.backends.locmem.LocMemCache",
    "django.core.cache.backends.dummy.DummyCache",
)


def check_webhook_dedup_cache(app_configs: Any = None, **kwargs: Any) -> list[Any]:
    """Warn when webhooks dedup events in a cache other processes cannot see."""
    from django.conf import settings

    from django_ai_sdk.integrations.registry import _registry
    from django_ai_sdk.integrations.webhooks.base import WebhookIntegration

    configured = getattr(settings, "AI_SDK_INTEGRATIONS", None) or {}
    webhooks = sorted(
        name
        for name, integration in _registry.items()
        if isinstance(integration, WebhookIntegration) and name in configured
    )
    backend = settings.CACHES.get("default", {}).get("BACKEND", "")
    if not webhooks or backend not in _UNSHARED_CACHES:
        return []
    return [
        Warning(
            f"Webhook integrations {', '.join(webhooks)} dedup events in the default cache, "
            f"which is {backend.rsplit('.', 1)[-1]}: not shared between processes, so a "
            "redelivered or replayed event can be answered twice.",
            hint="Use a shared cache (Redis, Memcached or DatabaseCache) as CACHES['default'].",
            id=f"{PREFIX}.W001",
        )
    ]


def dict_setting_errors(
    name: str, accepts: dict[str, tuple[Callable[[Any], bool], str]], prefix: str
) -> list[Error]:
    """E012: setting `name` is a dict of `accepts` keys, each of the kind it takes."""
    from django.conf import settings

    user_settings = getattr(settings, name, None) or {}
    if not isinstance(user_settings, dict):
        return [
            Error(
                f"{name} must be a dict, not a {type(user_settings).__name__}.",
                hint="Set a dict, or remove it.",
                id=f"{prefix}.E012",
            )
        ]
    errors = []
    for key, value in user_settings.items():
        if key not in accepts:
            errors.append(
                Error(
                    f"{name} has an unknown key {key!r}.",
                    hint=f"The keys are {', '.join(accepts)}.",
                    id=f"{prefix}.E012",
                )
            )
        elif not accepts[key][0](value):
            errors.append(Error(f"{name}[{key!r}] must be {accepts[key][1]}.", id=f"{prefix}.E012"))
    return errors


def _asks_for_policy(cls: type) -> bool:
    from django_ai_sdk.agents.base import Agent

    if getattr(cls, "oversight", None) not in (None, False):
        return True
    return getattr(cls, "get_oversight", None) is not Agent.get_oversight


__all__ = ["check_oversight_installed", "check_webhook_dedup_cache", "dict_setting_errors"]
