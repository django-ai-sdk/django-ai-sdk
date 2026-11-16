"""A WorkflowRun's lifecycle: open, queue, claim, record, close.

Runs execute on DBOS (`workflows.durable`); the lifecycle is told in the workflows manual.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from asgiref.sync import sync_to_async
from django.db import IntegrityError, transaction
from django.db.models import F
from django.db.models.functions import Coalesce
from django.utils import timezone
from pydantic import ValidationError

from django_ai_sdk.errors import UserError
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
    """Create a pending run of `workflow` with a copy of its definition.

    The `workflow_key` is set here. It is the stored workflow's slug, the registered
    workflow's name, or "adhoc" for any other definition. A trigger passes a `run_id`
    derived from what fired it, so firing twice opens one run.
    """
    if record is not None:
        key = record.slug
    elif is_registered(workflow):
        key = workflow.name
    else:
        key = ADHOC
    return await WorkflowRun.objects.acreate(
        **({"id": run_id} if run_id is not None else {}),
        trigger=trigger,
        workflow=record,
        workflow_key=key,
        workflow_definition=workflow.model_dump(),
        inputs=serialize(inputs or {}),
        user_id=user_pk(user),
        started_by_id=user_pk(started_by),
    )


def enqueue_run(run: WorkflowRun) -> None:
    """Queue `run` for a worker once the current transaction commits.

    On commit, so a worker never looks for a row that does not exist yet. If
    queueing fails, the run is marked failed with the reason, so it never sits
    pending forever.
    """
    run_id = str(run.pk)
    transaction.on_commit(lambda: requeue(run_id))


async def aenqueue_run(run: WorkflowRun) -> None:
    """`enqueue_run`, from async code. Joins the caller's transaction when there is one."""
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
    """Check the inputs, then create the run and queue it.

    `user` is who the run acts as; `started_by` the person who asked for it, when
    a person did. Neither may approve it.

    The inputs are checked here rather than in the worker, so a caller who supplied
    the wrong ones hears about it on the call that queued the run. A `run_id` that
    exists returns that run and queues nothing.
    """
    # Only what the definition declares is stored: the schema is the contract.
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
        # Fired twice at once: the other one opened it.
        return await WorkflowRun.objects.aget(pk=run_id)
    await aenqueue_run(run)
    return run


def requeue(run_id: str) -> None:
    """Hand the run's current attempt to DBOS. Idempotent: the workflow id is fixed per attempt.

    The worker calls it at start for every pending run, so a run whose web process
    died between commit and queueing is not stranded.
    """
    from django_ai_sdk.workflows.durable import enqueue

    try:
        task_id = enqueue(run_id)
    except Exception as exc:
        logger.exception("Workflow run %s could not be queued", run_id)
        WorkflowRun.objects.filter(pk=run_id, status=WorkflowRun.Status.PENDING).update(
            status=WorkflowRun.Status.FAILED,
            # The broker's own message can hold its URL; the log has it.
            error=f"Could not be queued ({type(exc).__name__}); see the server log.",
            completed_at=timezone.now(),
        )
        return
    WorkflowRun.objects.filter(pk=run_id).update(task_id=task_id)


def requeue_pending() -> int:
    """`requeue` every pending run. The worker calls it at start; returns how many."""
    pending = list(
        WorkflowRun.objects.filter(status=WorkflowRun.Status.PENDING).values_list("pk", flat=True)
    )
    for run_id in pending:
        requeue(str(run_id))
    return len(pending)


async def claim_run(run_id: str, *, workflow_id: str | None = None) -> WorkflowRun | None:
    """Claim a pending run for this worker. None when it is not pending.

    One conditional UPDATE, so a duplicate delivery or a finished run claims nothing.
    `workflow_id` is the DBOS workflow claiming it, stored as `task_id`. A run already
    running under that id is this workflow's own: it claimed the run and its worker
    died before DBOS recorded the claim, so the claim is taken up again, not refused.
    """
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
    """Claim `run_id`, walk its steps and close it. None when there was nothing to claim.

    `workflows.durable` calls this with DBOS's `perform`, `wait` and workflow id, so
    every unit is a checkpoint. The defaults do the same walk here and now, as tests
    and inline callers do; such a run cannot wait for an approval.
    """
    claimed = await perform("claim", lambda: _claim(run_id, workflow_id))
    if claimed is None:
        logger.info("Workflow run %s is not pending; nothing to claim", run_id)
        return None
    run = await WorkflowRun.objects.aget(pk=run_id)
    # The attempt this call claimed, whatever the row says by now.
    run.attempt = attempt = claimed["attempt"]
    try:
        await walk_run(run, completed=claimed["completed"], perform=perform, wait=wait)
    except RunSuperseded:
        raise
    except Exception as exc:
        # Failed before the walk began, or while the recorder closed the run.
        error = str(exc)
        await perform("fail", lambda: fail_run(run, attempt, error))
        raise
    return run


async def _claim(run_id: str, workflow_id: str | None) -> dict[str, Any] | None:
    run = await claim_run(run_id, workflow_id=workflow_id)
    if run is None:
        return None
    return {"attempt": run.attempt, "completed": serialize(await _already_completed(run))}


async def walk_run(
    run: WorkflowRun,
    *,
    completed: Mapping[str, Any],
    perform: Perform = perform_now,
    wait: Wait | None = None,
) -> None:
    """Walk a claimed run's steps under its current attempt. The recorder closes it."""
    from django_ai_sdk.tasks import aget_principal

    # The definition crossed the queue as JSON on the run row.
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
        # The key set when the run opened, not the name the definition gives itself.
        workflow=run.workflow_key,
        run_id=str(run.pk),
        perform=perform,
        wait=wait,
    )


async def fail_run(run: WorkflowRun, attempt: int, error: str) -> None:
    """Mark the run failed with `error`, unless the recorder already closed it."""
    await _close(run, attempt, status=WorkflowRun.Status.FAILED, error=error, quiet=True)


async def _close(run: WorkflowRun, attempt: int, *, quiet: bool = False, **fields: Any) -> None:
    """Write the run's ending, if this attempt still holds it."""
    fields["completed_at"] = timezone.now()
    closed = await WorkflowRun.objects.held(run.pk, attempt).aupdate(**fields)
    if closed:
        for name, value in fields.items():
            setattr(run, name, value)
    elif not quiet:
        logger.info("Workflow run %s attempt %s no longer holds the run", run.pk, attempt)


class RunRecorder(RunListener):
    """Write one StepRun row per step, and stop the walk once the run is not ours.

    Passed to `run_steps` as `recorder=`, so its errors stop the run. It raises
    `RunSuperseded` once the run is no longer running under its attempt.
    """

    def __init__(self, run: WorkflowRun, steps: Sequence[Step]) -> None:
        # Built by the worker, not a definition: it takes a run, not a config.
        super().__init__()
        self.run = run
        self.attempt = run.attempt
        self.sequence = {step.name: index for index, step in enumerate(steps)}
        self.closed = False

    async def beat(self) -> None:
        """Mark the run alive, or raise `RunSuperseded` once it is no longer ours (cancelled)."""
        alive = await WorkflowRun.objects.held(self.run.pk, self.attempt).aupdate(
            heartbeat_at=timezone.now()
        )
        if not alive:
            raise RunSuperseded(
                f"run {self.run.pk} is no longer running under attempt {self.attempt}"
            )

    async def on_run_end(self, ctx: WorkflowContext, error: BaseException | None) -> None:
        """Write the run's ending. The runner calls this before any listener's."""
        if isinstance(error, RunSuperseded):
            return  # The run belongs to a newer attempt.
        fields: dict[str, Any]
        if error is None:
            fields = {"status": WorkflowRun.Status.COMPLETED, "outputs": _published(ctx.steps)}
        else:
            fields = {"status": WorkflowRun.Status.FAILED, "error": str(error)}
        await _close(self.run, self.attempt, **fields)
        self.closed = True

    async def on_step_start(self, ctx: WorkflowContext, step: Step) -> None:
        """Claim this step's row for this attempt.

        The run claim already keeps other workers out, so a row left running by an
        earlier attempt is taken over. A completed row never is: the walk skips the
        steps earlier attempts completed, so a completed row here is this attempt's
        own, written before its worker died and DBOS recorded the step. It raises
        `StepRecorded` with what the row holds, and the step does not run again.
        """
        await self.beat()
        sequence = self.sequence[step.name]
        # ON CONFLICT DO NOTHING leaves an existing row for the update below, and
        # does not abort the caller's transaction.
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

    async def on_step_end(self, ctx: WorkflowContext, step: Step, outcome: StepOutcome) -> None:
        """Write a step's outcome. Upserts, because a skipped step never started."""
        await self.beat()
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
                # A waiting step has not ended.
                "completed_at": None if outcome.status == "waiting" else timezone.now(),
            },
        )


def validate_inputs(workflow: WorkflowDefinition, supplied: dict[str, Any]) -> dict[str, Any]:
    """The run's inputs, coerced to what the definition declares.

    A definition that declares no inputs accepts anything, so a pipeline written in
    code need not describe itself in JSON.
    """
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
    # Drop undeclared extras. The schema is the contract.
    return {name: getattr(parsed, name) for name in model.model_fields}


async def _already_completed(run: WorkflowRun) -> dict[str, Any]:
    """Stored output of every step of this run that already finished."""
    return {
        row.step_name: row.output
        async for row in StepRun.objects.filter(run=run, status=StepRun.Status.COMPLETED)
    }


def _published(outcomes: Mapping[str, StepOutcome]) -> dict[str, Any]:
    """What the run produced, keyed by step name."""
    return {
        name: serialize(outcome.output)
        for name, outcome in outcomes.items()
        if outcome.status == "completed"
    }


__all__ = [
    "aenqueue_run",
    "claim_run",
    "enqueue_run",
    "execute_run",
    "fail_run",
    "RunRecorder",
    "open_run",
    "queue_run",
    "requeue",
    "requeue_pending",
    "validate_inputs",
    "walk_run",
]
