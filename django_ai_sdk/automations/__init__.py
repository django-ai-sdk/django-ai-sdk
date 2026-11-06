"""Scheduled workflows: run a registered workflow on a clock.

An opt-in app on top of the workflows app::

    INSTALLED_APPS = [..., "django_ai_sdk.workflows", "django_ai_sdk.automations"]

An automation names a workflow, a cron schedule and the account it runs as. The
SDK dispatches due automations as WorkflowRuns. The deployment runs the tick
(`manage.py automations_tick`, from cron or a loop) and a django-tasks worker.
Declare them in any installed app's `automations.py`::

    from django_ai_sdk.automations import Automation, register_automation

    @register_automation
    class MorningDigest(Automation):
        name = "morning-digest"
        cron = "0 7 * * mon-fri"
        timezone = "Europe/Amsterdam"
        workflow = "digest"
        run_as = "digest-bot@example.com"
        input = "Summarise the open tasks since {last_run_at}."
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from django_ai_sdk.automations.base import Automation
from django_ai_sdk.automations.registry import (
    get_automation,
    get_automations,
    get_invalid_automations,
    register_automation,
    reset_registry,
)
from django_ai_sdk.automations.schedule import Cron

# These need the app registry, so they are imported on first access.
_LAZY = {
    "AutomationService": "django_ai_sdk.automations.services",
    "AutomationState": "django_ai_sdk.automations.models",
    "AutomationRun": "django_ai_sdk.automations.models",
}


def __getattr__(name: str) -> Any:
    if module := _LAZY.get(name):
        return getattr(import_module(module), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Automation",
    "AutomationRun",
    "AutomationService",
    "AutomationState",
    "Cron",
    "get_automation",
    "get_automations",
    "get_invalid_automations",
    "register_automation",
    "reset_registry",
]
