"""Queue entry point for workflow runs."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from asgiref.sync import async_to_sync
from django_tasks import task

from django_ai_sdk.workflows.models import WorkflowRun
from django_ai_sdk.workflows.schemas import WorkflowDefinition
from django_ai_sdk.workflows.steps import StepAlreadyRunning, StepFailed

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _recorded_outcomes_stay_off_the_queue(what: str) -> AsyncIterator[None]:
    """Swallow the two endings the run's own rows already describe.

    A stopped step and a refused claim are outcomes, not task failures: the rows say
    what happened. Anything else propagates, so the queue records what nothing else did.
    """
    try:
        yield
    except StepAlreadyRunning as exc:
        # Dispatch is at-least-once, so a second delivery finding the step claimed
        # is the ordinary case.
        logger.info("%s is already running at %s", what, exc)
    except StepFailed as exc:
        # The step's own row already says which one failed and why.
        logger.warning("%s stopped: %s", what, exc)


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
    async with _recorded_outcomes_stay_off_the_queue(f"Workflow run {run.id}"):
        await WorkflowExecutor().run(workflow, inputs=dict(run.inputs or {}), workflow_run=run)
