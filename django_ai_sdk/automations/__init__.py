"""Scheduled workflows: run agent work on a clock, for a resolved audience.

An automation says when it runs, as whom, and which workflow. The SDK claims and
dispatches due work; the deployment provides the tick (`manage.py run_automations`) and a
django-tasks worker. Declare them in any installed app's `automations.py`.

    from django_ai_sdk.automations import Audience, Automation, register

    @register
    class MorningDigest(Automation):
        name = "morning-digest"
        cron = "0 7 * * 1-5"              # UTC
        audience = Audience.SUBSCRIBED    # one run per opted-in user
        workflow = "digest"
        input = "Summarise my open tasks since {last_run_at}."
"""

from __future__ import annotations

from .audience import Audience, AudienceResolver
from .base import Automation
from .models import AutomationRun, AutomationState, AutomationSubscription
from .registry import (
    get_automation,
    get_automations,
    get_invalid_automations,
    register,
    reset_registry,
)
from .schedule import Cron
from .schemas import AutomationOut, AutomationRunOut
from .services import AutomationService

__all__ = [
    "Audience",
    "AudienceResolver",
    "Automation",
    "AutomationOut",
    "AutomationRun",
    "AutomationRunOut",
    "AutomationService",
    "AutomationState",
    "AutomationSubscription",
    "Cron",
    "get_automation",
    "get_automations",
    "get_invalid_automations",
    "register",
    "reset_registry",
]
