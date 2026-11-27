from __future__ import annotations

import logging
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from asgiref.sync import sync_to_async
from django.db import IntegrityError, transaction
from django.db.models import F, Q
from django.db.models.functions import Coalesce
from django.utils import timezone
from pydantic import ValidationError

from django_ai_sdk.errors import UserError, public_detail
from django_ai_sdk.permissions import user_pk
from django_ai_sdk.utils import serialize
from django_ai_sdk.workflows.definitions import compile_inputs, compile_listeners, compile_steps
from django_ai_sdk.workflows.listeners import RunListener
from django_ai_sdk.workflows.models import StepRun, WorkflowRun
from django_ai_sdk.workflows.registry import ADHOC, is_registered
from django_ai_sdk.workflows.runner import perform_now, run_steps
from django_ai_sdk.workflows.schemas import WorkflowDefinition
from django_ai_sdk.workflows.steps import RunSuperseded, StepOutcome, StepRecorded

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping, Sequence

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.workflows.models import Workflow
    from django_ai_sdk.workflows.runner import Perform, Wait
    from django_ai_sdk.workflows.steps import Step, WorkflowContext

logger = logging.getLogger(__name__)


async def open_run(
    workflow: WorkflowDefinition,
    *,
    inputs: dict[str, Any] | None = None,
    user: AbstractBaseUser | AnonymousUser | None = None,
    record: Workflow | None = None,
    run_id: uuid.UUID | None = None,
    trigger: str = "manual",
    started_by: AbstractBaseUser | AnonymousUser | None = None,
) -> WorkflowRun:
    if record is not None:
        key = record.slug
    elif is_registered(workflow):
        key = workflow.name
    else:
        key = ADHOC
    run = WorkflowRun(
        trigger=trigger,
        workflow=record,
        workflow_key=key,
        workflow_definition=workflow.model_dump(),
        inputs=serialize(inputs or {}),
        user_id=user_pk(user),
        started_by_id=user_pk(started_by),
    )
    if run_id is not None:
        run.id = run_id

    def create() -> WorkflowRun:
        with transaction.atomic():
            run.save(force_insert=True)
        return run

    return await sync_to_async(create)()


def enqueue_run(run: WorkflowRun) -> None:
    """Queue `run` on commit; if queueing fails, the run is marked failed."""
    run_id = str(run.pk)
    transaction.on_commit(lambda: requeue(run_id))


async def aenqueue_run(run: WorkflowRun) -> None:
    await sync_to_async(enqueue_run)(run)


async def queue_run(
    workflow: WorkflowDefinition,
    *,
    inputs: dict[str, Any] | None = None,
    user: AbstractBaseUser | AnonymousUser | None = None,
    record: Workflow | None = None,
    run_id: uuid.UUID | None = None,
    trigger: str = "manual",
    started_by: AbstractBaseUser | AnonymousUser | None = None,
) -> WorkflowRun:
    inputs = validate_inputs(workflow, inputs or {})
    if run_id is not None and (existing := await WorkflowRun.objects.filter(pk=run_id).afirst()):
        return existing
    try:
        run = await open_run(
            workflow,
            inputs=inputs,
            user=user,
            record=record,
            run_id=run_id,
            trigger=trigger,
            started_by=started_by,
        )
    except IntegrityError:
        if run_id is None:
            raise
        return await WorkflowRun.objects.aget(pk=run_id)
    await aenqueue_run(run)
    return run


def requeue(run_id: str) -> None:
    from django_ai_sdk.workflows.durable import enqueue

    try:
        task_id = enqueue(run_id)
    except Exception as exc:
        logger.exception("Workflow run %s could not be queued", run_id)
        WorkflowRun.objects.filter(pk=run_id, status=WorkflowRun.Status.PENDING).update(
            status=WorkflowRun.Status.FAILED,
            error=f"Could not be queued ({type(exc).__name__}); see the server log.",
            completed_at=timezone.now(),
        )
        return
    WorkflowRun.objects.filter(pk=run_id).update(task_id=task_id)


def requeue_pending() -> int:
    pending = list(
        WorkflowRun.objects.filter(status=WorkflowRun.Status.PENDING).values_list("pk", flat=True)
    )
    for run_id in pending:
        requeue(str(run_id))
    return len(pending)


def fail_stale_runs() -> int:
    """Fail stale running runs (and their open steps) that DBOS no longer runs; return count."""
    from django_ai_sdk.durable import runtime
    from django_ai_sdk.workflows.conf import setting
    from django_ai_sdk.workflows.durable import ACTIVE_STATUSES

    now = timezone.now()
    cutoff = now - timedelta(seconds=setting("STALE_AFTER_SECONDS"))
    stale = list(
        WorkflowRun.objects.filter(status=WorkflowRun.Status.RUNNING)
        .filter(Q(heartbeat_at__lt=cutoff) | Q(heartbeat_at__isnull=True))
        .values_list("pk", "attempt", "task_id")
    )
    if not stale:
        return 0
    ids = [task_id for _, _, task_id in stale if task_id]
    workflows = (
        runtime.client().list_workflows(workflow_ids=ids, load_input=False, load_output=False)
        if ids
        else []
    )
    running = {w.workflow_id for w in workflows if str(w.status) in ACTIVE_STATUSES}
    failed = 0
    for pk, attempt, task_id in stale:
        if task_id in running:
            continue
        with transaction.atomic():
            if not WorkflowRun.objects.held(pk, attempt).update(
                status=WorkflowRun.Status.FAILED,
                error="Stopped: its worker no longer runs it. Resume it to go on.",
                completed_at=now,
            ):
                continue
            StepRun.objects.filter(
                run_id=pk, status__in=(StepRun.Status.RUNNING, StepRun.Status.WAITING)
            ).update(status=StepRun.Status.FAILED, detail="its worker stopped", completed_at=now)
        logger.warning("Workflow run %s had no worker (DBOS workflow %s); failed it", pk, task_id)
        failed += 1
    return failed


async def claim_run(run_id: str, *, workflow_id: str | None = None) -> WorkflowRun | None:
    now = timezone.now()
    claimed = await WorkflowRun.objects.filter(
        pk=run_id, status=WorkflowRun.Status.PENDING
    ).aupdate(
        status=WorkflowRun.Status.RUNNING,
        attempt=F("attempt") + 1,
        heartbeat_at=now,
        started_at=Coalesce(F("started_at"), now),
        **({"task_id": workflow_id} if workflow_id else {}),
    )
    if not claimed and workflow_id:
        claimed = await WorkflowRun.objects.filter(
            pk=run_id, status=WorkflowRun.Status.RUNNING, task_id=workflow_id
        ).aupdate(heartbeat_at=now)
    if not claimed:
        return None
    return await WorkflowRun.objects.aget(pk=run_id)


async def execute_run(
    run_id: str,
    *,
    perform: Perform = perform_now,
    wait: Wait | None = None,
    workflow_id: str | None = None,
) -> WorkflowRun | None:
    """None when there was nothing to claim.

    With the default `perform`/`wait` it runs inline and cannot wait for an approval.
    """
    claimed = await perform("claim", lambda: _claim(run_id, workflow_id))
    if claimed is None:
        logger.info("Workflow run %s is not pending; nothing to claim", run_id)
        return None
    run = await WorkflowRun.objects.aget(pk=run_id)
    run.attempt = attempt = claimed["attempt"]
    try:
        await walk_run(run, completed=claimed["completed"], perform=perform, wait=wait)
    except RunSuperseded:
        raise
    except Exception as exc:
        error = public_detail(exc)
        await perform(
            "fail",
            lambda: _close(run, attempt, status=WorkflowRun.Status.FAILED, error=error, quiet=True),
        )
        raise
    return run


async def _claim(run_id: str, workflow_id: str | None) -> dict[str, Any] | None:
    run = await claim_run(run_id, workflow_id=workflow_id)
    if run is None:
        return None
    completed = {
        row.step_name: row.output
        async for row in StepRun.objects.filter(run=run, status=StepRun.Status.COMPLETED)
    }
    return {"attempt": run.attempt, "completed": serialize(completed)}


async def walk_run(
    run: WorkflowRun,
    *,
    completed: Mapping[str, Any],
    perform: Perform = perform_now,
    wait: Wait | None = None,
) -> None:
    from django_ai_sdk.tasks import aget_principal

    workflow = WorkflowDefinition.model_validate(run.workflow_definition)
    user = await aget_principal(run.user_id, source=f"Workflow run {run.pk}")
    steps = compile_steps(workflow)
    await run_steps(
        steps,
        inputs=validate_inputs(workflow, dict(run.inputs or {})),
        user=user,
        listeners=compile_listeners(workflow.listeners, workflow.name or str(run.pk)),
        recorder=RunRecorder(run, steps),
        completed=completed,
        workflow=run.workflow_key,
        run_id=str(run.pk),
        perform=perform,
        wait=wait,
    )


async def _close(run: WorkflowRun, attempt: int, *, quiet: bool = False, **fields: Any) -> None:
    fields["completed_at"] = timezone.now()
    closed = await WorkflowRun.objects.held(run.pk, attempt).aupdate(**fields)
    if closed:
        for name, value in fields.items():
            setattr(run, name, value)
    elif not quiet:
        logger.info("Workflow run %s attempt %s no longer holds the run", run.pk, attempt)


class RunRecorder(RunListener):
    """Write one StepRun row per step; raise `RunSuperseded` once another attempt owns the run."""

    def __init__(self, run: WorkflowRun, steps: Sequence[Step]) -> None:
        super().__init__()
        self.run = run
        self.attempt = run.attempt
        self.sequence = {step.name: index for index, step in enumerate(steps)}
        self.closed = False

    async def beat(self) -> None:
        """Mark the run alive, or raise `RunSuperseded` once it is no longer ours (cancelled)."""
        if not await self._alive():
            raise self._superseded()

    async def _alive(self) -> bool:
        return bool(
            await WorkflowRun.objects.held(self.run.pk, self.attempt).aupdate(
                heartbeat_at=timezone.now()
            )
        )

    def _superseded(self) -> RunSuperseded:
        return RunSuperseded(f"run {self.run.pk} is no longer running under attempt {self.attempt}")

    async def on_run_end(self, ctx: WorkflowContext, error: BaseException | None) -> None:
        """The runner calls this before any listener's."""
        if isinstance(error, RunSuperseded):
            return
        fields: dict[str, Any]
        if error is None:
            fields = {"status": WorkflowRun.Status.COMPLETED, "outputs": _published(ctx.steps)}
        else:
            fields = {"status": WorkflowRun.Status.FAILED, "error": str(error)}
        await _close(self.run, self.attempt, **fields)
        self.closed = True

    async def on_step_start(self, ctx: WorkflowContext, step: Step) -> None:
        """A completed row means the worker died before DBOS recorded it: raise `StepRecorded`."""
        await self.beat()
        sequence = self.sequence[step.name]
        await StepRun.objects.abulk_create(
            [StepRun(run=self.run, sequence=sequence, step_name=step.name)],
            ignore_conflicts=True,
        )
        claimed = (
            await StepRun.objects.filter(run=self.run, sequence=sequence)
            .exclude(status=StepRun.Status.COMPLETED)
            .aupdate(
                step_name=step.name,
                status=StepRun.Status.RUNNING,
                started_at=timezone.now(),
                completed_at=None,
                output=None,
                error="",
                detail="",
            )
        )
        if not claimed:
            row = await StepRun.objects.aget(run=self.run, sequence=sequence)
            raise StepRecorded(
                StepOutcome(output=row.output, detail=row.detail, turn_id=row.turn_id)
            )

    async def expire(self, step: Step) -> bool:
        """End a timed-out wait; False when a decision won (same WAITING update as decide)."""
        rows = StepRun.objects.filter(run=self.run, sequence=self.sequence[step.name])
        if await rows.filter(status=StepRun.Status.WAITING).aupdate(
            status=StepRun.Status.FAILED,
            detail="no decision came in time",
            completed_at=timezone.now(),
        ):
            return True
        return not await rows.filter(status=StepRun.Status.RUNNING).aexists()

    async def on_step_end(self, ctx: WorkflowContext, step: Step, outcome: StepOutcome) -> None:
        """A stopped run still records a completed step, so resume skips it; a newer attempt's not."""
        if await self._alive():
            await self._write(step, outcome)
            return
        if (
            outcome.status == "completed"
            and await WorkflowRun.objects.filter(pk=self.run.pk, attempt=self.attempt)
            .exclude(status=WorkflowRun.Status.RUNNING)
            .aexists()
        ):
            await self._write(step, outcome)
        raise self._superseded()

    async def _write(self, step: Step, outcome: StepOutcome) -> None:
        await StepRun.objects.aupdate_or_create(
            run=self.run,
            sequence=self.sequence[step.name],
            defaults={
                "step_name": step.name,
                "status": StepRun.Status(outcome.status),
                "output": serialize(outcome.output),
                "error": outcome.detail if outcome.status == "failed" else "",
                "detail": outcome.detail[:255],
                "turn_id": outcome.turn_id,
                "completed_at": None if outcome.status == "waiting" else timezone.now(),
            },
        )


def validate_inputs(workflow: WorkflowDefinition, supplied: dict[str, Any]) -> dict[str, Any]:
    model = compile_inputs(workflow)
    if model is None:
        return dict(supplied)
    try:
        parsed = model.model_validate(supplied)
    except ValidationError as exc:
        raise UserError(
            f"Workflow {workflow.name or '<unnamed>'!r} was given inputs it does not "
            f"declare, or is missing ones it does: {exc}"
        ) from exc
    return {name: getattr(parsed, name) for name in model.model_fields}


def _published(outcomes: Mapping[str, StepOutcome]) -> dict[str, Any]:
    return {
        name: serialize(outcome.output)
        for name, outcome in outcomes.items()
        if outcome.status == "completed"
    }


__all__ = [
    "aenqueue_run",
    "enqueue_run",
    "execute_run",
    "fail_stale_runs",
    "queue_run",
    "RunRecorder",
    "validate_inputs",
]
