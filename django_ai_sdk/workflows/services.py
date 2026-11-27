from __future__ import annotations

import contextlib
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from asgiref.sync import async_to_sync, sync_to_async
from django.core.exceptions import ImproperlyConfigured
from django.db.models.functions import Now
from django.utils import timezone
from pydantic import ValidationError

from django_ai_sdk.durable import runtime
from django_ai_sdk.errors import ErrorCode, UserError
from django_ai_sdk.permissions import (
    Operation,
    PermissionDenied,
    PermissionDomain,
    PermissionsMixin,
    user_pk,
)
from django_ai_sdk.workflows.conf import setting
from django_ai_sdk.workflows.definitions import inputs_json_schema
from django_ai_sdk.workflows.listeners import get_listener_registry
from django_ai_sdk.workflows.registry import (
    ADHOC,
    afind,
    definition_of,
    is_registered,
    validate_definition,
)
from django_ai_sdk.workflows.runs import aenqueue_run, queue_run
from django_ai_sdk.workflows.schemas import CronTrigger
from django_ai_sdk.workflows.triggers import (
    FIRE_CRON,
    async_schedules,
    declared_workflows,
    refuse_run_as,
    schedule_name,
    start_cron,
)

logger = logging.getLogger(__name__)

RUN_BLOBS = ("workflow_definition", "inputs")

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.workflows.models import WorkflowRun
    from django_ai_sdk.workflows.schemas import WorkflowDefinition


def _validate_submitted(workflow: WorkflowDefinition) -> None:
    try:
        validate_definition(workflow)
    except ImproperlyConfigured as exc:
        raise UserError(str(exc)) from exc
    refuse_run_as(workflow)


async def _refuse_changed_workflow(run: WorkflowRun) -> None:
    declared, record = await afind(run.workflow_key)
    if record is not None:
        with contextlib.suppress(ImproperlyConfigured, ValidationError):
            declared = definition_of(record)
    if declared is None:
        raise UserError(f"Workflow {run.workflow_key!r} no longer runs.")
    if declared.model_dump() != run.workflow_definition:
        raise UserError(
            f"Workflow {run.workflow_key!r} changed since run {run.pk} started; start a new run."
        )


async def _resync_schedules(key: str) -> None:
    try:
        await async_schedules(key)
    except Exception:
        logger.exception("Could not sync workflow schedules; the worker syncs them on start")


class DecisionConflict(UserError):
    """The step is not waiting for a decision: decided already, out of time, or stopped."""

    code = ErrorCode.CONFLICT


class WorkflowService(PermissionsMixin):
    domain = PermissionDomain.WORKFLOW

    @classmethod
    async def run(
        cls,
        workflow: WorkflowDefinition,
        *,
        inputs: dict[str, Any] | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> WorkflowRun:
        """For trusted code, never a client's JSON.

        Unregistered definitions also need MANAGE_WORKFLOW.
        """
        await cls.has_perms(user, Operation.RUN_WORKFLOW, raise_on_deny=True)
        if not is_registered(workflow):
            await cls.has_perms(user, Operation.MANAGE_WORKFLOW, raise_on_deny=True)
        _validate_submitted(workflow)
        return await queue_run(workflow, inputs=inputs, user=user, started_by=user)

    @classmethod
    async def start(
        cls,
        key: str,
        *,
        inputs: dict[str, Any] | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> WorkflowRun:
        """Queue a run of the workflow named `key` (registered wins over stored).

        Raises `Workflow.DoesNotExist` when missing or the caller may not run it.
        """
        from django_ai_sdk.workflows.models import Workflow

        await cls.has_perms(user, Operation.RUN_WORKFLOW, raise_on_deny=True)
        workflow, record = await afind(key)
        if record is not None and await cls.has_perms(
            user, Operation.RUN_WORKFLOW, obj=record, raise_on_deny=False
        ):
            try:
                workflow = definition_of(record)
            except (ImproperlyConfigured, ValidationError) as exc:
                logger.warning("Stored workflow %r cannot run: %s", key, exc)
                raise UserError(f"Workflow {key!r} cannot run; see the server log.") from exc
        if workflow is None:
            raise Workflow.DoesNotExist
        return await queue_run(workflow, inputs=inputs, user=user, record=record, started_by=user)

    @classmethod
    async def resume(
        cls,
        run_id: str,
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> WorkflowRun:
        """Queue a failed or cancelled run again, skipping completed steps.

        Ad-hoc runs also need MANAGE_WORKFLOW, as starting one did.
        """
        from django_ai_sdk.workflows.models import WorkflowRun

        run = await cls._own_run(run_id, user)
        if run.workflow_key == ADHOC:
            await cls.has_perms(user, Operation.MANAGE_WORKFLOW, raise_on_deny=True)
        else:
            await _refuse_changed_workflow(run)
        reopened = await WorkflowRun.objects.filter(
            pk=run.pk,
            status__in=(
                WorkflowRun.Status.FAILED,
                WorkflowRun.Status.CANCELLED,
            ),
        ).aupdate(
            status=WorkflowRun.Status.PENDING,
            error="",
            completed_at=None,
            task_id=None,
            updated_at=Now(),
        )
        if not reopened:
            raise UserError(
                f"Run {run.pk} is {run.status}; only a failed or cancelled run resumes."
            )
        await run.arefresh_from_db()
        await aenqueue_run(run)
        return run

    @classmethod
    async def cancel(
        cls,
        run_id: str,
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> WorkflowRun:
        """Stop a pending or running run; the step in flight finishes and keeps its outcome."""
        from django_ai_sdk.workflows.models import StepRun, WorkflowRun

        run = await cls._own_run(run_id, user)
        waiting = [
            name
            async for name in StepRun.objects.filter(
                run=run, status=StepRun.Status.WAITING
            ).values_list("step_name", flat=True)
        ]
        stopped = await WorkflowRun.objects.filter(
            pk=run.pk, status__in=(WorkflowRun.Status.PENDING, WorkflowRun.Status.RUNNING)
        ).aupdate(
            status=WorkflowRun.Status.CANCELLED,
            error="Cancelled.",
            completed_at=timezone.now(),
            updated_at=Now(),
        )
        if not stopped:
            raise UserError(f"Run {run.pk} is {run.status}; only a pending or running run stops.")
        await StepRun.objects.filter(
            run=run, status__in=(StepRun.Status.RUNNING, StepRun.Status.WAITING)
        ).aupdate(status=StepRun.Status.SKIPPED, detail="cancelled", completed_at=timezone.now())
        if run.task_id:
            # DBOS cancel does not wake a run blocked in recv; an empty decision does.
            for name in waiting:
                await runtime.send(
                    run.task_id,
                    None,
                    topic=f"approval:{name}",
                    idempotency_key=f"{run.task_id}:{name}:cancel",
                )
            await runtime.cancel(run.task_id)
        await run.arefresh_from_db()
        return run

    @classmethod
    async def decide(
        cls,
        run_id: str,
        step: str,
        *,
        approved: bool,
        note: str = "",
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> WorkflowRun:
        """Never by the run's own user or starter. First decision wins; later ones raise
        `DecisionConflict`.
        """
        from django_ai_sdk.workflows.models import StepRun, WorkflowRun

        await cls.has_perms(user, Operation.APPROVE_WORKFLOW, raise_on_deny=True)
        run = await WorkflowRun.objects.filter(id=run_id).afirst()
        if run is None or not await cls.has_perms(
            user, Operation.VIEW_WORKFLOW, run, raise_on_deny=False
        ):
            raise WorkflowRun.DoesNotExist
        await cls.has_perms(user, Operation.APPROVE_WORKFLOW, run, raise_on_deny=True)
        if not run.task_id:
            raise UserError(f"Run {run.pk} is not queued yet.")
        by = str(getattr(user, "get_username", lambda: "")() or user_pk(user) or "")
        claimed = await StepRun.objects.filter(
            run=run, step_name=step, status=StepRun.Status.WAITING
        ).aupdate(status=StepRun.Status.RUNNING, detail=f"decided by {by}"[:255])
        if not claimed:
            raise DecisionConflict(f"Step {step!r} of run {run.pk} is not waiting for a decision.")
        decision = {"approved": bool(approved), "by": by, "note": note[:1000]}
        try:
            await runtime.send(
                run.task_id,
                decision,
                topic=f"approval:{step}",
                idempotency_key=f"{run.task_id}:{step}",
            )
        except Exception:
            await StepRun.objects.filter(run=run, step_name=step).aupdate(
                status=StepRun.Status.WAITING, detail=""
            )
            raise
        await run.arefresh_from_db()
        return run

    @classmethod
    async def list_triggers(
        cls,
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> list[dict[str, Any]]:
        """Every declared trigger, with its schedule's state and its latest run."""
        from django_ai_sdk.workflows.models import WorkflowRun

        await cls.has_perms(user, Operation.VIEW_WORKFLOW, raise_on_deny=True)
        shows_run_as = await cls.has_perms(user, Operation.MANAGE_WORKFLOW, raise_on_deny=False)
        schedules = await _schedules()
        runs = await sync_to_async(cls.has_queryset_perms)(
            user, Operation.VIEW_WORKFLOW, queryset=WorkflowRun.objects.defer(*RUN_BLOBS)
        )
        found: list[dict[str, Any]] = []
        for key, definition, record in await sync_to_async(declared_workflows)():
            for trigger in definition.triggers:
                last = (
                    await runs.filter(workflow_key=key, trigger=f"{trigger.kind}:{trigger.name}")
                    .order_by("-created_at")
                    .afirst()
                )
                schedule = schedules.get(schedule_name(key, trigger.name))
                found.append(
                    {
                        "workflow": key,
                        "workflow_id": str(record.pk) if record else None,
                        "name": trigger.name,
                        "kind": trigger.kind,
                        "cron": getattr(trigger, "cron", ""),
                        "timezone": getattr(trigger, "timezone", ""),
                        "event": getattr(trigger, "event", ""),
                        "run_as": (
                            trigger.run_as or (str(record.created_by_id or "") if record else "")
                        )
                        if shows_run_as
                        else "",
                        "enabled": trigger.enabled,
                        "paused": bool(schedule and schedule["status"] != "ACTIVE"),
                        "next_run_at": _next_run(trigger, schedule),
                        "last_fired_at": schedule["last_fired_at"] if schedule else None,
                        "last_run": last,
                    }
                )
        return found

    @classmethod
    async def set_trigger_paused(
        cls,
        key: str,
        trigger: str,
        *,
        paused: bool,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> None:
        """A deploy keeps the choice."""
        name = await cls._managed_schedule(key, trigger, user)
        dbos = runtime.client()
        await sync_to_async(dbos.pause_schedule if paused else dbos.resume_schedule)(name)

    @classmethod
    async def fire_trigger(
        cls,
        key: str,
        trigger: str,
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> None:
        """The caller is recorded as starter (so cannot approve). Stored runs act as their
        author, so only the author fires them by hand.
        """
        await cls._managed_schedule(key, trigger, user)
        _declared, record = await afind(key)
        if record is not None and str(record.created_by_id or "") != str(user_pk(user) or ""):
            raise PermissionDenied("Only the workflow's author fires its trigger by hand.")
        await start_cron(key, trigger, timezone.now().isoformat(), started_by=user)

    @classmethod
    async def _managed_schedule(
        cls, key: str, trigger: str, user: AbstractBaseUser | AnonymousUser | None
    ) -> str:
        from django_ai_sdk.workflows.models import Workflow

        await cls.has_perms(user, Operation.MANAGE_WORKFLOW, raise_on_deny=True)
        declared, record = await afind(key)
        if record is not None:
            if not await cls.has_perms(
                user, Operation.MANAGE_WORKFLOW, obj=record, raise_on_deny=False
            ):
                raise Workflow.DoesNotExist
            declared = definition_of(record)
        if declared is None or not any(
            isinstance(t, CronTrigger) and t.name == trigger and t.enabled
            for t in declared.triggers
        ):
            raise Workflow.DoesNotExist
        return schedule_name(key, trigger)

    @classmethod
    async def _own_run(
        cls, run_id: str, user: AbstractBaseUser | AnonymousUser | None
    ) -> WorkflowRun:
        from django_ai_sdk.workflows.models import WorkflowRun

        await cls.has_perms(user, Operation.RUN_WORKFLOW, raise_on_deny=True)
        run = await WorkflowRun.objects.filter(id=run_id).afirst()
        if run is None or not await cls.has_perms(
            user, Operation.RUN_WORKFLOW, run, raise_on_deny=False
        ):
            raise WorkflowRun.DoesNotExist
        return run

    @classmethod
    async def list_listeners(
        cls,
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> list[dict[str, str]]:
        await cls.has_perms(user, Operation.MANAGE_WORKFLOW, raise_on_deny=True)
        return [
            {"key": key, "description": listener.description}
            for key, listener in get_listener_registry().items()
        ]

    @classmethod
    async def get_inputs_schema(
        cls,
        workflow_id: str,
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> dict[str, Any] | None:
        """The stored workflow's inputs as a JSON Schema; None if the caller may not see it."""
        from django_ai_sdk.workflows.models import Workflow

        await cls.has_perms(user, Operation.VIEW_WORKFLOW, raise_on_deny=True)
        record = await Workflow.objects.filter(id=workflow_id, active=True).afirst()
        if record is None or not await cls.has_perms(
            user, Operation.VIEW_WORKFLOW, obj=record, raise_on_deny=False
        ):
            return None
        return inputs_json_schema(definition_of(record))

    @classmethod
    async def create(
        cls,
        name: str,
        workflow: WorkflowDefinition,
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> Any:
        from django_ai_sdk.workflows.models import Workflow

        await cls.has_perms(user, Operation.MANAGE_WORKFLOW, raise_on_deny=True)
        _validate_submitted(workflow)

        record = Workflow(
            name=name,
            definition=workflow.model_dump(),
            created_by_id=user_pk(user),
        )
        await record.asave()
        await _resync_schedules(record.slug)
        return record

    @classmethod
    async def get(
        cls,
        workflow_id: str,
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> Any:
        """The stored definition this caller may manage, raising DoesNotExist otherwise."""
        from django_ai_sdk.workflows.models import Workflow

        record = await Workflow.objects.filter(id=workflow_id).afirst()
        if record is None or not await cls.has_perms(
            user, Operation.MANAGE_WORKFLOW, obj=record, raise_on_deny=False
        ):
            raise Workflow.DoesNotExist
        return record

    @classmethod
    async def update(
        cls,
        workflow_id: str,
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
        name: str | None = None,
        workflow: WorkflowDefinition | None = None,
        active: bool | None = None,
    ) -> Any:
        record = await cls.get(workflow_id, user=user)
        if workflow is not None:
            _validate_submitted(workflow)
            record.definition = workflow.model_dump()
            record.created_by_id = user_pk(user)
        if name is not None:
            record.name = name
        if active is not None:
            if active and not record.active:
                record.created_by_id = user_pk(user)
            record.active = active
        await record.asave()
        await _resync_schedules(record.slug)
        return record

    @classmethod
    async def delete(
        cls,
        workflow_id: str,
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> None:
        from django_ai_sdk.workflows.models import Workflow

        record = await cls.get(workflow_id, user=user)
        await Workflow.objects.filter(id=record.id).adelete()
        await _resync_schedules(record.slug)

    @classmethod
    async def list_workflows(
        cls,
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
        active_only: bool = True,
        limit: int | None = 100,
        offset: int = 0,
    ) -> list[Any]:
        from django_ai_sdk.workflows.models import Workflow

        await cls.has_perms(user, Operation.MANAGE_WORKFLOW, raise_on_deny=True)
        qs = Workflow.objects.all()
        if active_only:
            qs = qs.filter(active=True)
        qs = await sync_to_async(cls.has_queryset_perms)(
            user, Operation.MANAGE_WORKFLOW, queryset=qs
        )
        return [r async for r in qs[offset : offset + limit if limit is not None else None]]

    @classmethod
    async def list_runs(
        cls,
        workflow_id: str,
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
        limit: int | None = 50,
        offset: int = 0,
    ) -> list[Any]:
        from django_ai_sdk.workflows.models import WorkflowRun

        await cls.has_perms(user, Operation.VIEW_WORKFLOW, raise_on_deny=True)
        qs = (
            WorkflowRun.objects.filter(workflow_id=workflow_id)
            .defer(*RUN_BLOBS)
            .order_by("-created_at")
        )
        qs = await sync_to_async(cls.has_queryset_perms)(user, Operation.VIEW_WORKFLOW, queryset=qs)
        return [r async for r in qs[offset : offset + limit if limit is not None else None]]

    @classmethod
    async def get_run(
        cls,
        run_id: str,
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
        workflow_id: str | None = None,
    ) -> Any | None:
        """One run with its steps, or None. With `workflow_id`, only a run of that workflow."""
        from django_ai_sdk.workflows.models import WorkflowRun

        runs = WorkflowRun.objects.filter(id=run_id)
        if workflow_id is not None:
            runs = runs.filter(workflow_id=workflow_id)
        run = await runs.prefetch_related("steps").afirst()
        if run is None:
            return None
        if not await cls.has_perms(user, Operation.VIEW_WORKFLOW, run, raise_on_deny=False):
            return None
        return run


async def _schedules() -> dict[str, Any]:
    return {
        schedule["schedule_name"]: schedule
        for schedule in await runtime.client().list_schedules_async()
        if schedule["workflow_name"] == FIRE_CRON
    }


def _next_run(trigger: Any, schedule: dict[str, Any] | None) -> str | None:
    if (
        not isinstance(trigger, CronTrigger)
        or schedule is None
        or schedule["status"] != "ACTIVE"
        or not setting("SCHEDULES_ENABLED")
    ):
        return None
    from croniter import croniter

    now = datetime.now(ZoneInfo(trigger.timezone))
    return croniter(trigger.cron, now).get_next(datetime).astimezone(UTC).isoformat()


start_workflow = async_to_sync(WorkflowService.start)
resume_workflow_run = async_to_sync(WorkflowService.resume)
cancel_workflow_run = async_to_sync(WorkflowService.cancel)
decide_workflow_step = async_to_sync(WorkflowService.decide)
list_workflow_triggers = async_to_sync(WorkflowService.list_triggers)
set_workflow_trigger_paused = async_to_sync(WorkflowService.set_trigger_paused)
fire_workflow_trigger = async_to_sync(WorkflowService.fire_trigger)
list_workflow_listeners = async_to_sync(WorkflowService.list_listeners)
create_workflow = async_to_sync(WorkflowService.create)
update_workflow = async_to_sync(WorkflowService.update)
delete_workflow = async_to_sync(WorkflowService.delete)
get_workflow = async_to_sync(WorkflowService.get)
list_workflows = async_to_sync(WorkflowService.list_workflows)
list_workflow_runs = async_to_sync(WorkflowService.list_runs)
get_workflow_run = async_to_sync(WorkflowService.get_run)

__all__ = ["DecisionConflict", "WorkflowService"]
