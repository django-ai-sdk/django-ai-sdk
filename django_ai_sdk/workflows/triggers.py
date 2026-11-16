"""What starts a run besides a person asking: a cron schedule, or an event.

A workflow declares its triggers (`schemas.CronTrigger`, `schemas.EventTrigger`).

- Cron: `sync_schedules()` keeps one DBOS schedule per cron trigger, named
  "<workflow>:<trigger>". DBOS fires it once per occurrence across all workers,
  and `workflows.durable.fire_cron` opens the run. An operator pauses and resumes a schedule at
  runtime; a deploy keeps that choice. AI_SDK_WORKFLOWS['SCHEDULES_ENABLED'] off makes
  every occurrence start nothing, without touching the schedules.
- Event: every webhook delivery is "<integration>.message" (`start_for_webhook`);
  the host reports others with `start_for_event(event, payload, idempotency_key=...)`.
  Each enabled trigger on that event opens a run.

Either way the run's id comes from what fired it (the occurrence, or the
idempotency key), so firing twice opens one run.

Who a run acts as: `run_as` on a workflow declared in code, or whoever last wrote a
stored workflow's steps (`created_by`). A stored definition cannot name an account,
and editing one makes the editor its account, or authoring a workflow would be a
way to act as anyone.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import TYPE_CHECKING, Any

from asgiref.sync import async_to_sync
from django.core.exceptions import ImproperlyConfigured
from pydantic import ValidationError

from django_ai_sdk.errors import UserError
from django_ai_sdk.workflows.registry import afind, definition_of, get_declared_workflows
from django_ai_sdk.workflows.schemas import CronTrigger, EventTrigger

if TYPE_CHECKING:
    from dbos import ClientScheduleInput
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.integrations.webhooks.base import InboundEvent
    from django_ai_sdk.workflows.models import Workflow, WorkflowRun
    from django_ai_sdk.workflows.schemas import WorkflowDefinition

logger = logging.getLogger(__name__)

# The DBOS workflow every cron schedule fires.
FIRE_CRON = "ai_sdk.trigger_cron"
# Run ids derive from this namespace and what fired the run.
NAMESPACE = uuid.UUID("4f3a8e52-8f0e-4c55-9d5e-2f6a1d0c7b91")

# Bare `{word}` only: str.format would reach attributes and choke on prose.
_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def schedule_name(key: str, trigger: str) -> str:
    return f"{key}:{trigger}"


def fill(template: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    """`template` with each "{word}" filled from `values`. "{word}" alone keeps the value's type."""

    def one(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        if (whole := _PLACEHOLDER.fullmatch(value)) and whole.group(1) in values:
            return values[whole.group(1)]
        return _PLACEHOLDER.sub(
            lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0), value
        )

    return {name: one(value) for name, value in template.items()}


# --- who and what ----------------------------------------------------------------


def declared_workflows() -> list[tuple[str, WorkflowDefinition, Workflow | None]]:
    """(key, definition, row) for every workflow that can fire: code ones, then active rows.

    A row shadowed by a code workflow of the same name is left out, as at start. A
    row that cannot run is logged and left out.
    """
    from django_ai_sdk.workflows.models import Workflow

    declared = get_declared_workflows()
    found: list[tuple[str, WorkflowDefinition, Workflow | None]] = [
        (key, definition, None) for key, definition in declared.items()
    ]
    for row in Workflow.objects.filter(active=True).exclude(slug__in=list(declared)):
        try:
            found.append((row.slug, definition_of(row), row))
        except (ImproperlyConfigured, ValidationError) as exc:
            logger.warning(
                "Stored workflow %r cannot run, so its triggers are off: %s", row.slug, exc
            )
    return found


async def run_user(
    trigger: CronTrigger | EventTrigger, record: Workflow | None
) -> AbstractBaseUser | None:
    """Who a triggered run acts as: `run_as` in code, or who last wrote a stored workflow's steps."""
    from django_ai_sdk.tasks import aget_principal
    from django_ai_sdk.utils import aget_active_user

    if record is not None:
        return await aget_principal(record.created_by_id, source=f"Workflow {record.slug}")
    if trigger.run_as:
        user = await aget_active_user(trigger.run_as)
        if user is None:
            raise UserError(
                f"Trigger {trigger.name!r} runs as {trigger.run_as!r}, who is not active."
            )
        return user
    return None


def refuse_run_as(definition: WorkflowDefinition) -> None:
    """Raise UserError when a definition someone submitted names an account to act as."""
    if named := sorted(trigger.name for trigger in definition.triggers if trigger.run_as):
        raise UserError(
            f"Triggers {named} set run_as. Only a workflow declared in code acts as another "
            "account; a stored workflow's runs act as the person who created it."
        )


# --- cron -------------------------------------------------------------------------


def cron_schedules(only: str | None = None) -> dict[str, ClientScheduleInput]:
    """The DBOS schedule each enabled cron trigger should have, by schedule name.

    `only` limits it to the workflow with that key.
    """
    from django_ai_sdk.workflows.durable import QUEUE

    schedules: dict[str, ClientScheduleInput] = {}
    for key, definition, _ in declared_workflows():
        if only is not None and key != only:
            continue
        for trigger in definition.triggers:
            if isinstance(trigger, CronTrigger) and trigger.enabled:
                name = schedule_name(key, trigger.name)
                schedules[name] = {
                    "schedule_name": name,
                    "workflow_name": FIRE_CRON,
                    "schedule": trigger.cron,
                    "context": {"key": key, "trigger": trigger.name},
                    "cron_timezone": trigger.timezone,
                    "queue_name": QUEUE,
                }
    return schedules


def sync_schedules(only: str | None = None) -> list[str]:
    """Make the DBOS schedules match the declared cron triggers. Returns their names.

    New and changed ones are written, and ones no longer declared (or switched off
    in the definition) are deleted. A schedule an operator paused stays paused.
    `only` touches just that workflow's schedules, so two web processes saving
    different workflows at once cannot delete each other's.
    """
    from django_ai_sdk.durable.runtime import client

    wanted = cron_schedules(only)
    dbos = client()
    if wanted:
        dbos.apply_schedules(list(wanted.values()))
    for existing in dbos.list_schedules():
        if (
            existing["workflow_name"] == FIRE_CRON
            and existing["schedule_name"] not in wanted
            and (only is None or (existing["context"] or {}).get("key") == only)
        ):
            dbos.delete_schedule(existing["schedule_name"])
    return sorted(wanted)


async def async_schedules(only: str | None = None) -> list[str]:
    """`sync_schedules`, from async code, after a stored workflow changed."""
    from asgiref.sync import sync_to_async

    return await sync_to_async(sync_schedules)(only)


async def start_cron(
    key: str,
    trigger_name: str,
    scheduled_at: str,
    *,
    started_by: AbstractBaseUser | AnonymousUser | None = None,
) -> str | None:
    """Open and queue the run for one occurrence. Returns its id, or None when nothing runs.

    `started_by` is the person who fired it by hand (`WorkflowService.fire_trigger`).
    """
    from django_ai_sdk.workflows.models import WorkflowRun
    from django_ai_sdk.workflows.runs import queue_run

    found = await _find_trigger(key, trigger_name, CronTrigger)
    if found is None:
        return None
    definition, record, trigger = found
    last = (
        await WorkflowRun.objects.filter(workflow_key=key, trigger=f"cron:{trigger.name}")
        .order_by("-created_at")
        .values_list("created_at", flat=True)
        .afirst()
    )
    inputs = fill(
        trigger.inputs,
        {
            "name": trigger.name,
            "workflow": key,
            "scheduled_at": scheduled_at,
            "last_run_at": last.isoformat() if last else "the beginning",
        },
    )
    run = await queue_run(
        definition,
        inputs=inputs,
        user=await run_user(trigger, record),
        record=record,
        run_id=uuid.uuid5(NAMESPACE, f"cron:{key}:{trigger.name}:{scheduled_at}"),
        trigger=f"cron:{trigger.name}",
        started_by=started_by,
    )
    return str(run.pk)


# --- events -----------------------------------------------------------------------


async def start_for_event(
    event: str, payload: dict[str, Any], *, idempotency_key: str
) -> list[WorkflowRun]:
    """Start a run of every workflow with an enabled trigger on `event`.

    `idempotency_key` names this delivery, such as a message's id: delivered again,
    it starts nothing new and returns the runs it started the first time. A
    trigger whose inputs the workflow refuses is logged and skipped.
    """
    from django_ai_sdk.workflows.runs import queue_run

    started: list[WorkflowRun] = []
    for key, definition, record in await _declared():
        for trigger in definition.triggers:
            if not (
                isinstance(trigger, EventTrigger) and trigger.enabled and trigger.event == event
            ):
                continue
            try:
                run = await queue_run(
                    definition,
                    inputs=fill(trigger.inputs, payload),
                    user=await run_user(trigger, record),
                    record=record,
                    run_id=uuid.uuid5(NAMESPACE, f"event:{key}:{trigger.name}:{idempotency_key}"),
                    trigger=f"event:{trigger.name}",
                )
            except UserError as exc:
                logger.warning("Event %r did not start %s/%s: %s", event, key, trigger.name, exc)
                continue
            started.append(run)
    return started


start_for_event_sync = async_to_sync(start_for_event)


async def start_for_webhook(sender: Any, event: InboundEvent, **kwargs: Any) -> None:
    """Report a webhook delivery as "<integration>.message". Never fails the webhook.

    Connected to `event_received`. The event id is the idempotency key, so a
    redelivery starts nothing new.
    """
    if not event.event_id:
        return
    try:
        await start_for_event(
            f"{event.integration}.message",
            {"text": event.text, "conversation_id": event.conversation_id},
            idempotency_key=f"{event.integration}:{event.event_id}",
        )
    except Exception:
        logger.exception("Could not start triggers for %s %s", event.integration, event.event_id)


async def _declared() -> list[tuple[str, WorkflowDefinition, Workflow | None]]:
    from asgiref.sync import sync_to_async

    return await sync_to_async(declared_workflows)()


async def _find_trigger(
    key: str, name: str, kind: type[CronTrigger] | type[EventTrigger]
) -> tuple[WorkflowDefinition, Workflow | None, Any] | None:
    declared, record = await afind(key)
    try:
        definition = declared if record is None else definition_of(record)
    except (ImproperlyConfigured, ValidationError) as exc:
        logger.warning("Workflow %r cannot run, so trigger %r did not fire: %s", key, name, exc)
        return None
    if definition is None:
        logger.warning("Trigger %r fired for workflow %r, which no longer exists", name, key)
        return None
    for trigger in definition.triggers:
        if isinstance(trigger, kind) and trigger.name == name and trigger.enabled:
            return definition, record, trigger
    logger.info("Trigger %r on workflow %r is gone or off; nothing started", name, key)
    return None


__all__ = [
    "FIRE_CRON",
    "async_schedules",
    "cron_schedules",
    "declared_workflows",
    "fill",
    "refuse_run_as",
    "run_user",
    "schedule_name",
    "start_cron",
    "start_for_event",
    "start_for_event_sync",
    "start_for_webhook",
    "sync_schedules",
]
