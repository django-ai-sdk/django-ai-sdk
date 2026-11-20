"""A WorkflowRun's lifecycle: open, queue, claim, record, close.

The lifecycle (claim, heartbeat, sweep, resume) is told in the workflows manual.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Any

from asgiref.sync import sync_to_async
from django.db import transaction
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
from django_ai_sdk.workflows.runner import run_steps
from django_ai_sdk.workflows.schemas import WorkflowDefinition
from django_ai_sdk.workflows.steps import RunSuperseded

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Mapping, Sequence

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.workflows.models import Workflow
    from django_ai_sdk.workflows.steps import Step, StepOutcome, WorkflowContext

logger = logging.getLogger(__name__)

# How often a worker beats while a step runs. Keep it well under `workflows_sweep --stale-after`.
HEARTBEAT_SECONDS = 60


async def open_run(
    workflow: WorkflowDefinition,
    *,
    inputs: dict[str, Any] | None = None,
    user: AbstractBaseUser | AnonymousUser | None = None,
    record: Workflow | None = None,
) -> WorkflowRun:
    """Create a pending run of `workflow` with a copy of its definition.

    The `workflow_key` is set here. It is the stored workflow's slug, the registered
    workflow's name, or "adhoc" for any other definition.
    """
    if record is not None:
        key = record.slug
    elif is_registered(workflow):
        key = workflow.name
    else:
        key = ADHOC
    return await WorkflowRun.objects.acreate(
        workflow=record,
        workflow_key=key,
        workflow_definition=workflow.model_dump(),
        inputs=serialize(inputs or {}),
        user_id=user_pk(user),
    )


def enqueue_run(run: WorkflowRun) -> None:
    """Queue `run` for a worker once the current transaction commits.

    On commit, so a worker never looks for a row that does not exist yet. If
    queueing fails, the run is marked failed with the reason, so it never sits
    pending forever.
    """
    run_id = str(run.pk)
    transaction.on_commit(lambda: _enqueue(run_id))


async def aenqueue_run(run: WorkflowRun) -> None:
    """`enqueue_run`, from async code. Joins the caller's transaction when there is one."""
    await sync_to_async(enqueue_run)(run)


async def queue_run(
    workflow: WorkflowDefinition,
    *,
    inputs: dict[str, Any] | None = None,
    user: AbstractBaseUser | AnonymousUser | None = None,
    record: Workflow | None = None,
) -> WorkflowRun:
    """Check the inputs, then create the run and queue it.

    The inputs are checked here rather than in the worker, so a caller who supplied
    the wrong ones hears about it on the call that queued the run.
    """
    # Only what the definition declares is stored: the schema is the contract.
    inputs = validate_inputs(workflow, inputs or {})
    run = await open_run(workflow, inputs=inputs, user=user, record=record)
    await aenqueue_run(run)
    return run


def _enqueue(run_id: str) -> None:
    from django_ai_sdk.workflows.tasks import execute_workflow

    try:
        task = execute_workflow.enqueue(run_id)
    except Exception as exc:
        logger.exception("Workflow run %s could not be queued", run_id)
        WorkflowRun.objects.filter(pk=run_id, status=WorkflowRun.Status.PENDING).update(
            status=WorkflowRun.Status.FAILED,
            # The broker's own message can hold its URL; the log has it.
            error=f"Could not be queued ({type(exc).__name__}); see the server log.",
            completed_at=timezone.now(),
        )
        return
    WorkflowRun.objects.filter(pk=run_id).update(task_id=task.id)


async def claim_run(run_id: str) -> WorkflowRun | None:
    """Claim a pending run for this worker. None when it is not pending.

    One conditional UPDATE, so a duplicate delivery or a finished run claims nothing.
    """
    now = timezone.now()
    claimed = await WorkflowRun.objects.filter(
        pk=run_id, status=WorkflowRun.Status.PENDING
    ).aupdate(
        status=WorkflowRun.Status.RUNNING,
        attempt=F("attempt") + 1,
        heartbeat_at=now,
        started_at=Coalesce(F("started_at"), now),
    )
    if not claimed:
        return None
    return await WorkflowRun.objects.aget(pk=run_id)


async def execute_run(run_id: str) -> WorkflowRun | None:
    """Claim `run_id`, walk its steps and close it. None when there was nothing to claim.

    The recorder closes the run before the listeners hear it ended. Any error marks
    the run failed and propagates, except `RunSuperseded`, which leaves the run to
    the newer attempt that owns it.
    """
    from django_ai_sdk.tasks import aget_principal

    run = await claim_run(run_id)
    if run is None:
        logger.info("Workflow run %s is not pending; nothing to claim", run_id)
        return None
    attempt = run.attempt
    recorder: RunRecorder | None = None
    try:
        # The definition crossed the queue as JSON on the run row.
        workflow = WorkflowDefinition.model_validate(run.workflow_definition)
        user = await aget_principal(run.user_id, source=f"Workflow run {run.pk}")
        steps = compile_steps(workflow)
        recorder = RunRecorder(run, steps)
        async with recorder.beating():
            await run_steps(
                steps,
                inputs=validate_inputs(workflow, dict(run.inputs or {})),
                user=user,
                listeners=compile_listeners(workflow.listeners, workflow.name or str(run.pk)),
                recorder=recorder,
                completed=await _already_completed(run),
                # The key set when the run opened, not the name the definition gives itself.
                workflow=run.workflow_key,
                run_id=str(run.pk),
            )
    except RunSuperseded:
        raise
    except Exception as exc:
        # Failed before the walk began, or while the recorder closed the run.
        if recorder is None or not recorder.closed:
            await _close(run, attempt, status=WorkflowRun.Status.FAILED, error=str(exc))
        raise
    return run


async def _close(run: WorkflowRun, attempt: int, **fields: Any) -> None:
    """Write the run's ending, if this attempt still holds it."""
    fields["completed_at"] = timezone.now()
    closed = await WorkflowRun.objects.held(run.pk, attempt).aupdate(**fields)
    if closed:
        for name, value in fields.items():
            setattr(run, name, value)
    else:
        logger.info("Workflow run %s attempt %s no longer holds the run", run.pk, attempt)


class RunRecorder(RunListener):
    """Write one StepRun row per step and keep the run's heartbeat.

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
        """Show the run is alive, or raise `RunSuperseded` if it is no longer ours."""
        alive = await WorkflowRun.objects.held(self.run.pk, self.attempt).aupdate(
            heartbeat_at=timezone.now()
        )
        if not alive:
            raise RunSuperseded(
                f"run {self.run.pk} is no longer running under attempt {self.attempt}"
            )

    @contextlib.asynccontextmanager
    async def beating(self) -> AsyncGenerator[None]:
        """Beat every HEARTBEAT_SECONDS in the background, so a long step is not swept."""

        async def beat_forever() -> None:
            while True:
                await asyncio.sleep(HEARTBEAT_SECONDS)
                try:
                    await self.beat()
                except RunSuperseded:
                    return  # The next step boundary raises it in the run itself.
                except Exception:
                    # A missed beat must not fail the run; the next one may land.
                    logger.warning("Heartbeat for run %s failed", self.run.pk, exc_info=True)

        pulse = asyncio.create_task(beat_forever())
        try:
            yield
        finally:
            pulse.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pulse

    async def on_run_end(self, ctx: WorkflowContext, error: BaseException | None) -> None:
        """Write the run's ending. The runner calls this before any listener's."""
        if isinstance(error, RunSuperseded):
            return  # The run belongs to a newer attempt.
        if error is None:
            fields = {"status": WorkflowRun.Status.COMPLETED, "outputs": _published(ctx.steps)}
        else:
            fields = {"status": WorkflowRun.Status.FAILED, "error": str(error)}
        await _close(self.run, self.attempt, **fields)
        self.closed = True

    async def on_step_start(self, ctx: WorkflowContext, step: Step) -> None:
        """Claim this step's row for this attempt.

        The run claim already keeps other workers out, so a row left running by an
        earlier attempt is taken over. A completed row never is.
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
            raise RunSuperseded(
                f"step {step.name!r} (sequence {sequence}) of run {self.run.pk} "
                f"already completed under another attempt"
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
                "completed_at": timezone.now(),
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
    "RunRecorder",
    "open_run",
    "queue_run",
    "validate_inputs",
]
