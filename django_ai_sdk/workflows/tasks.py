"""Queue entry point for workflow runs."""

from __future__ import annotations

import logging

from asgiref.sync import async_to_sync
from django_tasks import task

from django_ai_sdk.tasks import aget_principal
from django_ai_sdk.workflows.models import WorkflowRun
from django_ai_sdk.workflows.schemas import WorkflowDefinition
from django_ai_sdk.workflows.steps import StepAlreadyRunning, StepFailed

logger = logging.getLogger(__name__)


@task(queue_name="default")
def execute_workflow(run_id: str) -> None:
    """Sync task entry point — worker calls this, bridges to async executor."""
    async_to_sync(_execute_async)(run_id)


async def _execute_async(run_id: str) -> None:
    """Load a stored run and execute its definition."""
    from django_ai_sdk.workflows.executor import WorkflowExecutor  # lazy — breaks circular

    run = await WorkflowRun.objects.aget(id=run_id)
    # The definition crossed the queue as JSON on the run row. Inputs stay JSON;
    # each step coerces what it reads.
    workflow = WorkflowDefinition.model_validate(run.workflow_definition)
    user = await aget_principal(run.user_id, source=f"Workflow run {run.id}")
    # A stopped step and a refused claim are outcomes, not task failures: the run's
    # own rows already say what happened. Anything else propagates.
    try:
        await WorkflowExecutor().run(
            workflow, inputs=dict(run.inputs or {}), user=user, workflow_run=run
        )
    except StepAlreadyRunning as exc:
        logger.info("Workflow run %s is already running at %s", run.id, exc)
    except StepFailed as exc:
        logger.warning("Workflow run %s stopped: %s", run.id, exc)
