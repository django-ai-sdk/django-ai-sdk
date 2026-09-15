"""Running a stored WorkflowDefinition and recording the result."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from django.utils import timezone
from pydantic import ValidationError

from django_ai_sdk.permissions import user_pk
from django_ai_sdk.utils import serialize
from django_ai_sdk.workflows.definitions import compile_hooks, compile_inputs, compile_steps
from django_ai_sdk.workflows.hooks import RunRecorder
from django_ai_sdk.workflows.models import WorkflowRun, WorkflowRunStep
from django_ai_sdk.workflows.runner import run_steps
from django_ai_sdk.workflows.steps import StepAlreadyRunning
from django_ai_sdk.workflows.tasks import execute_workflow

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.workflows.schemas import WorkflowDefinition
    from django_ai_sdk.workflows.steps import StepOutcome

logger = logging.getLogger(__name__)


class WorkflowExecutor:
    @staticmethod
    async def enqueue(run: WorkflowRun) -> None:
        """Schedule a WorkflowRun as a background task."""
        task = await execute_workflow.aenqueue(str(run.id))
        await WorkflowRun.objects.filter(id=run.id).aupdate(task_id=task.id)

    async def run(
        self,
        workflow: WorkflowDefinition,
        *,
        inputs: dict[str, Any] | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
        workflow_run: WorkflowRun | None = None,
    ) -> tuple[dict[str, Any], WorkflowRun]:
        """Compile the definition and run it, recording each step against the run."""
        if workflow_run is not None and workflow_run.status == WorkflowRun.Status.COMPLETED:
            return workflow_run.outputs or {}, workflow_run

        # Resuming: the row's persisted inputs sit underneath, the caller's overlay them.
        persisted = dict(workflow_run.inputs) if workflow_run and workflow_run.inputs else {}
        supplied = {**persisted, **(inputs or {})}

        workflow_run = await self._aopen(workflow, supplied, user, workflow_run)

        try:
            steps = compile_steps(workflow)
            hooks = [
                RunRecorder(workflow_run, steps),
                *compile_hooks(workflow.hooks, workflow.name or str(workflow_run.id)),
            ]
            outcomes = await run_steps(
                steps,
                inputs=validate_inputs(workflow, supplied),
                principal=user,
                hooks=hooks,
                completed=await _already_completed(workflow_run),
                workflow=workflow.name,
                run_id=str(workflow_run.id),
            )
            outputs = _published(outcomes)
        except StepAlreadyRunning:
            raise
        except Exception as exc:
            workflow_run.status = WorkflowRun.Status.FAILED
            workflow_run.error = str(exc)
            workflow_run.completed_at = timezone.now()
            await workflow_run.asave(
                update_fields=["status", "error", "completed_at", "updated_at"]
            )
            raise

        workflow_run.status = WorkflowRun.Status.COMPLETED
        workflow_run.outputs = outputs
        workflow_run.completed_at = timezone.now()
        await workflow_run.asave(update_fields=["status", "outputs", "completed_at", "updated_at"])
        return outputs, workflow_run

    @staticmethod
    async def _aopen(
        workflow: WorkflowDefinition,
        inputs: dict[str, Any],
        user: AbstractBaseUser | AnonymousUser | None,
        workflow_run: WorkflowRun | None,
    ) -> WorkflowRun:
        """The run row this attempt records against, opened or resumed."""
        if workflow_run is None:
            return await WorkflowRun.objects.acreate(
                workflow=None,
                workflow_definition=workflow.model_dump(),
                status=WorkflowRun.Status.RUNNING,
                inputs=serialize(inputs),
                user_id=user_pk(user),
                started_at=timezone.now(),
            )
        workflow_run.status = WorkflowRun.Status.RUNNING
        if not workflow_run.started_at:
            workflow_run.started_at = timezone.now()
        fields = ["status", "started_at", "updated_at"]
        if not workflow_run.inputs:
            workflow_run.inputs = serialize(inputs)
            fields.append("inputs")
        await workflow_run.asave(update_fields=fields)
        return workflow_run


def validate_inputs(workflow: WorkflowDefinition, supplied: dict[str, Any]) -> dict[str, Any]:
    """The run's inputs, coerced to what the definition declares.

    A definition that declares nothing takes what it is given: a code-authored
    pipeline is not obliged to describe itself in JSON.
    """
    model = compile_inputs(workflow)
    if model is None:
        return dict(supplied)
    try:
        parsed = model.model_validate(supplied)
    except ValidationError as exc:
        raise ValueError(
            f"Workflow {workflow.name or '<unnamed>'!r} was given inputs it does not "
            f"declare, or is missing ones it does: {exc}"
        ) from exc
    # Undeclared extras are dropped: the schema is the contract.
    return {name: getattr(parsed, name) for name in model.model_fields}


async def _already_completed(run: WorkflowRun) -> dict[str, Any]:
    """Stored output of every step of this run that already finished."""
    return {
        row.step_name: row.output
        async for row in run.steps.filter(status=WorkflowRunStep.Status.COMPLETED)
    }


def _published(outcomes: dict[str, StepOutcome]) -> dict[str, Any]:
    """What the run produced, keyed by step name."""
    return {
        name: serialize(outcome.output)
        for name, outcome in outcomes.items()
        if outcome.status == "completed"
    }


__all__ = ["WorkflowExecutor", "validate_inputs"]
