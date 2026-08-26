"""Due-ness, claiming, fan-out and dispatch: everything one tick does.

A tick is stateless and safe to run on any number of hosts: the claim is one conditional
UPDATE advancing the cursor and taking the lease together, so a second tick matches no
rows. Dispatch is at-least-once, so payloads must be idempotent.
"""

from __future__ import annotations

import logging
import uuid
from datetime import timedelta
from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple

from asgiref.sync import sync_to_async
from django.db import transaction
from django.db.models import Q
from django.utils import timezone

from django_ai_sdk.automations.config import automations_enabled, is_enabled, lease_seconds
from django_ai_sdk.automations.models import AutomationRun, AutomationState
from django_ai_sdk.automations.registry import get_automation, get_automations

if TYPE_CHECKING:
    from datetime import datetime

    from django_ai_sdk.automations.base import Automation

logger = logging.getLogger(__name__)

# Whether the kill-switch notice has been logged in this process.
_warned_disabled = False


class Dispatched(NamedTuple):
    """What one tick did to one automation."""

    name: str
    runs: list[AutomationRun]
    reason: str = ""


async def tick(
    *,
    now: datetime | None = None,
    only: str = "",
    force: bool = False,
    dry_run: bool = False,
) -> list[Dispatched]:
    """Claim and dispatch every automation that is due.

    `force` ignores due-ness but never the lease.
    """
    deliberate = bool(only and force)
    if not automations_enabled() and not deliberate:
        global _warned_disabled
        if not _warned_disabled:
            # Once per process; `--loop 60` would otherwise log a line a minute.
            _warned_disabled = True
            logger.info("AI_SDK_AUTOMATIONS_ENABLED is False; nothing dispatched.")
        return []

    now = now or timezone.now()
    automations = get_automations()
    if only:
        automations = {name: a for name, a in automations.items() if name == only}
        if not automations:
            return [Dispatched(only, [], "no automation with that name")]

    results: list[Dispatched] = []
    for name, automation in automations.items():
        try:
            results.append(
                await _dispatch_one(
                    automation, now=now, force=force, dry_run=dry_run, deliberate=deliberate
                )
            )
        except Exception:
            # One broken automation must not stop the rest of the tick.
            logger.exception("Automation %r failed to dispatch", name)
            results.append(Dispatched(name, [], "dispatch raised; see logs"))
    return results


async def _dispatch_one(
    automation: Automation,
    *,
    now: datetime,
    force: bool,
    dry_run: bool,
    deliberate: bool = False,
) -> Dispatched:
    name = automation.name
    state = await AutomationState.objects.filter(name=name).afirst()

    # Being off is a statement about the schedule, which a person naming it is not.
    if not deliberate:
        enabled, source = is_enabled(
            name, code_default=automation.enabled, db_value=state.enabled if state else None
        )
        if not enabled:
            return Dispatched(name, [], f"disabled ({source})")

    if state is None:
        # Nothing is written on a dry run, so first sight stays first sight.
        if dry_run:
            return Dispatched(name, [], "not due; the next real tick creates its schedule")
        state, _ = await ensure_state(automation, now=now)

    due_at = state.next_run_at
    if not force and due_at > now:
        return Dispatched(name, [], "not due")

    if dry_run:
        return Dispatched(name, [], "would dispatch (dry run)")

    if not await claim(automation, state, now=now, force=force):
        return Dispatched(name, [], "not claimed; another tick won it, or a run is in flight")

    principals = await _resolve_audience(automation)
    if not principals:
        run = await _record_skip(automation, state, now=now, reason="audience resolved to nobody")
        await release(state)
        return Dispatched(name, [run], "audience empty")

    # The occurrence it is for, not the moment it ran.
    scheduled_for = now if force else due_at
    dispatch_id = uuid.uuid4()
    runs = [
        await _create_run(automation, state, user=user, now=scheduled_for, dispatch_id=dispatch_id)
        for user in principals
    ]
    await _enqueue_all(runs)
    return Dispatched(name, runs)


async def ensure_state(automation: Automation, *, now: datetime) -> tuple[AutomationState, bool]:
    """Fetch this automation's state row, creating it on first sight.

    Scheduled forward, so deploying one does not fire it for a past cron match.
    """
    schedule = automation.get_schedule()
    return await AutomationState.objects.aget_or_create(
        name=automation.name,
        defaults={
            "next_run_at": schedule.next_after(now),
            "schedule_repr": str(schedule),
        },
    )


async def claim(
    automation: Automation, state: AutomationState, *, now: datetime, force: bool = False
) -> bool:
    """Win the right to dispatch this automation for this tick.

    Predicate and write are one statement, so the second tick's UPDATE matches no rows.
    """
    schedule = automation.get_schedule()
    next_at = schedule.next_after(now)
    lease_until = now + timedelta(seconds=lease_seconds())

    qs = AutomationState.objects.filter(id=state.id)
    if not force:
        qs = qs.filter(next_run_at__lte=now)
    if not automation.allow_overlap:
        qs = qs.filter(Q(locked_until__isnull=True) | Q(locked_until__lt=now))

    won = await qs.aupdate(
        next_run_at=next_at,
        last_dispatched_at=now,
        locked_until=lease_until,
        schedule_repr=str(schedule),
        updated_at=now,
    )
    if won:
        state.next_run_at = next_at
        state.locked_until = lease_until
    return bool(won)


async def release(state: AutomationState, *, succeeded_at: datetime | None = None) -> None:
    """Drop the lease this dispatch took, leaving a lease a later tick took alone.

    Predicated on the lease value, so a straggler cannot free a dispatch still in flight.
    """
    if succeeded_at is not None:
        await AutomationState.objects.filter(id=state.id).aupdate(last_success_at=succeeded_at)
    await AutomationState.objects.filter(id=state.id, locked_until=state.locked_until).aupdate(
        locked_until=None
    )


async def _resolve_audience(automation: Automation) -> list[Any]:
    try:
        return await automation.audience.resolve(automation)
    except Exception:
        logger.exception(
            "Automation %r could not resolve its audience; treating it as empty",
            automation.name,
        )
        return []


async def _create_run(
    automation: Automation,
    state: AutomationState,
    *,
    user: Any,
    now: datetime,
    dispatch_id: uuid.UUID,
    trigger: str = AutomationRun.Trigger.SCHEDULE,
) -> AutomationRun:
    return await AutomationRun.objects.acreate(
        name=automation.name,
        state=state,
        dispatch_id=dispatch_id,
        status=AutomationRun.Status.PENDING,
        trigger=trigger,
        user=user,
        scheduled_for=now,
    )


async def _record_skip(
    automation: Automation, state: AutomationState, *, now: datetime, reason: str
) -> AutomationRun:
    """Write a SKIPPED run: silence is indistinguishable from a scheduler that is down."""
    logger.info("Automation %r skipped: %s", automation.name, reason)
    return await AutomationRun.objects.acreate(
        name=automation.name,
        state=state,
        dispatch_id=uuid.uuid4(),
        status=AutomationRun.Status.SKIPPED,
        trigger=AutomationRun.Trigger.SCHEDULE,
        scheduled_for=now,
        finished_at=now,
        skip_reason=reason[:255],
    )


async def _enqueue_all(runs: list[AutomationRun]) -> None:
    """Hand each run to the task backend once its row is visible to a worker.

    Enqueued from on_commit, so a caller inside a transaction cannot hand a worker an id
    it would not find.
    """
    for run in runs:
        await sync_to_async(transaction.on_commit)(partial(_enqueue, run.id))


def _enqueue(run_id: uuid.UUID) -> None:
    """Enqueue one run and record its task id, or fail the run if the backend will not."""
    from django_ai_sdk.automations.tasks import execute_automation

    try:
        result = execute_automation.enqueue(str(run_id))
    except Exception:
        logger.exception("Could not enqueue automation run %s", run_id)
        AutomationRun.objects.filter(id=run_id).update(
            status=AutomationRun.Status.FAILED,
            error="Could not enqueue; is a task backend configured?",
            finished_at=timezone.now(),
        )
        return
    AutomationRun.objects.filter(id=run_id).update(task_id=str(result.id))


async def run_now(name: str) -> list[AutomationRun]:
    """Dispatch an automation immediately, outside its schedule.

    Resolves the audience as a tick does, so this fans out to every subscriber, not to
    whoever asked.
    """
    automation = get_automation(name)
    if automation is None:
        raise ValueError(f"No automation named {name!r}")

    now = timezone.now()
    state, _ = await ensure_state(automation, now=now)
    if not await claim(automation, state, now=now, force=True):
        raise RuntimeError(
            f"Automation {name!r} is already running (its lease is held). Wait for it "
            "to finish, or set allow_overlap = True if concurrent runs are safe."
        )

    principals = await _resolve_audience(automation)
    if not principals:
        run = await _record_skip(automation, state, now=now, reason="audience resolved to nobody")
        await release(state)
        return [run]

    dispatch_id = uuid.uuid4()
    runs = [
        await _create_run(
            automation,
            state,
            user=u,
            now=now,
            dispatch_id=dispatch_id,
            trigger=AutomationRun.Trigger.MANUAL,
        )
        for u in principals
    ]
    await _enqueue_all(runs)
    return runs


__all__ = [
    "Dispatched",
    "claim",
    "ensure_state",
    "release",
    "run_now",
    "tick",
]
