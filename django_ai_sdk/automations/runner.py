"""One scheduler tick: find due automations, claim them and dispatch their runs.

Ticks are safe to run on many hosts at once. The claim is one conditional UPDATE
that advances the cursor, so a second tick matches no rows. Each run is a
WorkflowRun, linked by an `AutomationRun` row and executed by the workflow worker.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

from django.utils import timezone

from django_ai_sdk.automations.models import AutomationRun, AutomationState
from django_ai_sdk.automations.registry import get_automation, get_automations
from django_ai_sdk.utils import app_setting

if TYPE_CHECKING:
    from datetime import datetime

    from django_ai_sdk.automations.base import Automation
    from django_ai_sdk.workflows.models import WorkflowRun

logger = logging.getLogger(__name__)

DEFAULTS = {
    # False dispatches nothing, scheduled or by hand.
    "ENABLED": True,
}


class Dispatched(NamedTuple):
    """What one tick did to one automation."""

    name: str
    run: WorkflowRun | None
    reason: str = ""


def _kill_switch_on() -> bool:
    return not app_setting("AI_SDK_AUTOMATIONS", DEFAULTS, "ENABLED")


async def tick(*, now: datetime | None = None) -> list[Dispatched]:
    """Claim and dispatch every automation that is due."""
    if _kill_switch_on():
        logger.info("AI_SDK_AUTOMATIONS['ENABLED'] is False; nothing dispatched.")
        return []

    now = now or timezone.now()
    results: list[Dispatched] = []
    for name, automation in get_automations().items():
        try:
            results.append(await _tick_one(automation, now=now))
        except Exception:
            # One broken automation must not stop the rest of the tick.
            logger.exception("Automation %r failed to dispatch", name)
            results.append(Dispatched(name, None, "dispatch raised; see logs"))
    return results


def resolve_enabled(automation: Automation, state: AutomationState | None) -> tuple[bool, str]:
    """Whether the schedule fires it, and which layer decided (kill-switch, db or code)."""
    if _kill_switch_on():
        return False, "kill-switch"
    if state is not None and state.enabled is not None:
        return state.enabled, "db"
    return automation.enabled, "code"


async def _tick_one(automation: Automation, *, now: datetime) -> Dispatched:
    existing = await AutomationState.objects.filter(name=automation.name).afirst()
    if not resolve_enabled(automation, existing)[0]:
        return Dispatched(automation.name, None, "disabled")

    state = await ensure_state(automation, now=now)
    if state.next_run_at is None or state.next_run_at > now:
        return Dispatched(automation.name, None, "not due")

    if not await _claim(automation, state, now=now):
        return Dispatched(automation.name, None, "another tick claimed it")

    return Dispatched(automation.name, await dispatch(automation))


async def ensure_state(automation: Automation, *, now: datetime | None = None) -> AutomationState:
    """Get or create this automation's state row.

    A new row is scheduled after `now`, so deploying does not fire a past occurrence.
    """
    state, _ = await AutomationState.objects.aget_or_create(
        name=automation.name,
        defaults={"next_run_at": automation.get_schedule().next_after(now or timezone.now())},
    )
    return state


async def _claim(automation: Automation, state: AutomationState, *, now: datetime) -> bool:
    """Claim this occurrence. True when this tick won it.

    The check and the write are one UPDATE, so a competing tick matches no rows.
    """
    next_at = automation.get_schedule().next_after(now)
    won = await AutomationState.objects.filter(id=state.id, next_run_at__lte=now).aupdate(
        next_run_at=next_at, last_dispatched_at=now, updated_at=now
    )
    if won:
        state.next_run_at = next_at
    return bool(won)


async def dispatch_now(name: str) -> WorkflowRun | None:
    """Dispatch an automation immediately, outside its schedule, enabled or not.

    Leaves `next_run_at` alone, so a pending scheduled occurrence still fires.
    """
    automation = get_automation(name)
    if automation is None:
        raise ValueError(f"No automation named {name!r}")
    if _kill_switch_on():
        logger.info("AI_SDK_AUTOMATIONS['ENABLED'] is False; %r not dispatched.", name)
        return None

    now = timezone.now()
    state = await ensure_state(automation, now=now)
    await AutomationState.objects.filter(id=state.id).aupdate(
        last_dispatched_at=now, updated_at=now
    )
    return await dispatch(automation)


async def dispatch(automation: Automation) -> WorkflowRun | None:
    """Open a WorkflowRun as `run_as` and queue it.

    Returns None and logs why when the run is skipped. A skip leaves no row.
    """
    from django_ai_sdk.utils import aget_active_user
    from django_ai_sdk.workflows.registry import afind, definition_of
    from django_ai_sdk.workflows.runs import queue_run

    # The registry's own lookup: code wins over an active stored row.
    definition, record = await afind(automation.workflow)
    if record is not None:
        definition = definition_of(record)
    if definition is None:
        logger.warning(
            "Automation %r runs workflow %r, which is not registered; skipped",
            automation.name,
            automation.workflow,
        )
        return None

    user = await aget_active_user(automation.run_as)
    # Never fall back to no user. The run would silently lose the account's access.
    if automation.run_as and user is None:
        logger.error(
            "Automation %r: run_as %r matches no single active user; skipped",
            automation.name,
            automation.run_as,
        )
        return None

    inputs = automation.get_inputs(user=user, last_run_at=await _last_run_at(automation))

    # A workflow drops undeclared inputs, so the run would succeed without its
    # prompt. Skip it instead.
    declared = definition.input_fields
    undeclared = sorted(set(inputs) - set(declared)) if declared else []
    if undeclared:
        logger.warning(
            "Automation %r supplies %s, which workflow %r does not declare; skipped. "
            "Declare them in its input_fields or override get_inputs().",
            automation.name,
            ", ".join(repr(key) for key in undeclared),
            automation.workflow,
        )
        return None
    # A stored workflow's runs point at its row, like WorkflowService's.
    run = await queue_run(definition, inputs=inputs, user=user, record=record)
    await AutomationRun.objects.acreate(run=run, automation=automation.name)
    return run


async def _last_run_at(automation: Automation) -> datetime | None:
    """When this automation's last completed run started."""
    from django_ai_sdk.workflows.models import WorkflowRun

    run = (
        await WorkflowRun.objects.filter(
            automation_run__automation=automation.name, status=WorkflowRun.Status.COMPLETED
        )
        .order_by("-created_at")
        .afirst()
    )
    if run is None:
        return None
    return run.started_at or run.created_at


__all__ = ["Dispatched", "dispatch", "dispatch_now", "tick"]
