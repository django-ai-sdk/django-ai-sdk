"""System checks for the core app. Ids are `ai_sdk.E001` and up."""

from __future__ import annotations

from typing import Any

from django.core.checks import Error

PREFIX = "ai_sdk"
OVERSIGHT_APP = "django_ai_sdk.oversight"


def check_oversight_installed(app_configs: Any = None, **kwargs: Any) -> list[Any]:
    """Report policies that are set while the oversight app is not installed.

    Without the app, those agents run unguarded. An agent wants a policy when it
    sets `oversight` or overrides `get_oversight()`. A DEFAULT_POLICY in
    AI_SDK_OVERSIGHT applies to every agent.
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


def _asks_for_policy(cls: type) -> bool:
    from django_ai_sdk.agents.base import Agent

    if getattr(cls, "oversight", None) not in (None, False):
        return True
    return getattr(cls, "get_oversight", None) is not Agent.get_oversight


__all__ = ["check_oversight_installed"]
