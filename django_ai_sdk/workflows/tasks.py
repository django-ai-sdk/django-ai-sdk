from __future__ import annotations

import logging

from asgiref.sync import async_to_sync
from django_tasks import task

from django_ai_sdk.workflows.steps import RunSuperseded, WorkflowStopped

logger = logging.getLogger(__name__)


@task(queue_name="default")
def execute_workflow(run_id: str) -> None:
    """Run one queued WorkflowRun in this worker. The queue carries only its id."""
    from django_ai_sdk.workflows.runs import execute_run  # lazy: runs reaches the task queue

    # A stopped run or a superseded attempt is an outcome, not a task failure. The
    # run's rows already record it. Anything else, including a step that raised,
    # fails the task.
    try:
        async_to_sync(execute_run)(run_id)
    except RunSuperseded as exc:
        logger.info("Workflow run %s was taken over: %s", run_id, exc)
    except WorkflowStopped as exc:
        logger.warning("Workflow run %s stopped: %s", run_id, exc)
