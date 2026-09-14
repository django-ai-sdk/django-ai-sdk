"""Running a stored WorkflowDefinition and recording the result."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from django.utils import timezone

from django_ai_sdk.permissions import user_pk
from django_ai_sdk.workflows.actions import ActionContext, get_action_registry
from django_ai_sdk.workflows.definitions import compile_steps
from django_ai_sdk.workflows.inputs import dump_messages, normalize_workflow_inputs
from django_ai_sdk.workflows.models import WorkflowRun
from django_ai_sdk.workflows.runner import run_steps
from django_ai_sdk.workflows.sink import WorkflowRunStepSink
from django_ai_sdk.workflows.tasks import execute_workflow

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.workflows.schemas import WorkflowDefinition
    from django_ai_sdk.workflows.steps import Step, StepOutcome

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
        messages: list[ChatMessage] | None = None,
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
        seeded = normalize_workflow_inputs(inputs=inputs, messages=messages)
        seeded = normalize_workflow_inputs(inputs={**persisted, **seeded}, ensure_messages=True)

        workflow_run = await self._aopen(workflow, seeded, user, workflow_run)

        try:
            steps = compile_steps(workflow)
            outcomes = await run_steps(
                steps,
                inputs=seeded,
                principal=user,
                sink=WorkflowRunStepSink(workflow_run, steps),
            )
            outputs = _published(steps, outcomes)
            await self._arun_actions(workflow, outputs, user=user, workflow_run=workflow_run)
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
        dumped_messages = dump_messages(inputs.get("messages"))
        persisted = {
            **{k: v for k, v in inputs.items() if k != "messages"},
            "messages": dumped_messages,
        }
        if workflow_run is None:
            return await WorkflowRun.objects.acreate(
                workflow=None,
                workflow_definition=workflow.model_dump(),
                status=WorkflowRun.Status.RUNNING,
                inputs=persisted,
                user_id=user_pk(user),
                started_at=timezone.now(),
            )
        workflow_run.status = WorkflowRun.Status.RUNNING
        if not workflow_run.started_at:
            workflow_run.started_at = timezone.now()
        if not workflow_run.inputs:
            workflow_run.inputs = persisted
            await workflow_run.asave(update_fields=["status", "started_at", "inputs", "updated_at"])
        else:
            await workflow_run.asave(update_fields=["status", "started_at", "updated_at"])
        return workflow_run

    @staticmethod
    async def _arun_actions(
        workflow: WorkflowDefinition,
        outputs: dict[str, Any],
        *,
        user: AbstractBaseUser | AnonymousUser | None,
        workflow_run: WorkflowRun,
    ) -> None:
        """Run each declared action over the run's outputs."""
        registry = get_action_registry()
        producer = {step.output_key: step.agent_id for step in workflow.steps}
        last_agent_id = workflow.steps[-1].agent_id if workflow.steps else ""
        source = workflow.name or f"workflow:{workflow_run.id}"
        for action in workflow.actions:
            runner_cls = registry.get(action.type)
            if runner_cls is None:
                logger.warning("Unknown workflow action type: %s", action.type)
                continue
            if action.input_key and action.input_key not in outputs:
                logger.warning(
                    "Workflow action %r reads %r, which the run did not produce — skipping",
                    action.type,
                    action.input_key,
                )
                continue
            payload = outputs.get(action.input_key) if action.input_key else outputs
            context = ActionContext(
                user=user,
                agent_id=producer.get(action.input_key or "", last_agent_id),
                source=source,
            )
            await runner_cls().execute(payload, context)
            logger.debug("Workflow action %r complete", action.type)


def _published(steps: list[Step], outcomes: dict[str, StepOutcome]) -> dict[str, Any]:
    """What the run produced, keyed by `provides` and `error_key` names."""
    published: dict[str, Any] = {}
    for step in steps:
        outcome = outcomes.get(step.name)
        if outcome is None:
            continue
        if outcome.status == "completed" and step.provides:
            published[step.provides] = outcome.output
        elif outcome.status == "failed" and step.error_key:
            published[step.error_key] = {
                "step": step.name,
                "error": outcome.detail or "failed",
            }
    return published
