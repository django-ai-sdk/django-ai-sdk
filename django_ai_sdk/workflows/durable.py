"""A workflow run as a DBOS workflow: each walk unit is a checkpointed step.

A recovered run replays finished units instead of redoing them; approvals wait with
`DBOS.recv`. Workflow id is the run id, or "<run id>:<attempt>" on resume.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from dbos import DBOS

from django_ai_sdk.durable import runtime
from django_ai_sdk.utils import serialize
from django_ai_sdk.workflows.models import WorkflowRun
from django_ai_sdk.workflows.steps import RunSuperseded, StepOutcome, WorkflowStopped
from django_ai_sdk.workflows.triggers import FIRE_CRON

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

RUN_WORKFLOW = "ai_sdk.workflow_run"
QUEUE = "workflows"


def workflow_id(run_id: str, attempt: int) -> str:
    """The DBOS workflow id for the next attempt at a run that has had `attempt` claims."""
    return run_id if attempt == 0 else f"{run_id}:{attempt + 1}"


def enqueue(run_id: str) -> str:
    """Queue the run's next attempt on DBOS. Returns the workflow id, also stored as task_id."""
    attempt = WorkflowRun.objects.filter(pk=run_id).values_list("attempt", flat=True).first()
    queued = workflow_id(run_id, attempt or 0)
    runtime.client().enqueue(
        {
            "queue_name": QUEUE,
            "workflow_name": RUN_WORKFLOW,
            "workflow_id": queued,
            "app_version": runtime.setting("VERSION"),
        },
        run_id,
    )
    return queued


# DBOS workflow statuses under which DBOS still runs, recovers or dequeues the workflow.
ACTIVE_STATUSES = frozenset({"PENDING", "ENQUEUED", "DELAYED", "SUSPENDED"})


def runs_under_other_versions() -> dict[str, str]:
    """Unfinished runs under another AI_SDK_DBOS['VERSION']: {run id: version}.

    Only a worker of that version picks them up; the worker warns about them at start.
    """
    current = runtime.setting("VERSION")
    unfinished = runtime.client().list_workflows(
        name=RUN_WORKFLOW,
        status=list(ACTIVE_STATUSES),
        load_input=False,
        load_output=False,
    )
    return {
        w.workflow_id.split(":")[0]: str(w.app_version)
        for w in unfinished
        if w.app_version != current
    }


@DBOS.workflow(name=RUN_WORKFLOW)
async def run_workflow(run_id: str) -> str:
    """Claim the run, walk its steps, close it. Returns how it ended."""
    from django_ai_sdk.workflows.runs import execute_run

    try:
        run = await execute_run(
            run_id, perform=perform, wait=DBOS.recv_async, workflow_id=DBOS.workflow_id
        )
    except (WorkflowStopped, RunSuperseded) as exc:
        # An outcome the rows already record, not a failure of the workflow.
        logger.info("Workflow run %s ended: %s", run_id, exc)
        return "stopped"
    return "not pending" if run is None else "done"


@DBOS.workflow(name=FIRE_CRON)
async def fire_cron(scheduled_at: str, context: dict[str, Any]) -> str | None:
    """What a cron schedule runs (`workflows.triggers`): open the occurrence's run, once."""
    from django_ai_sdk.workflows.conf import setting
    from django_ai_sdk.workflows.triggers import start_cron

    if not setting("SCHEDULES_ENABLED"):
        logger.info("Schedules are off; %s did not start a run", context)
        return None
    return await DBOS.run_step_async(
        {"name": "start"}, start_cron, context["key"], context["trigger"], scheduled_at
    )


async def perform(label: str, unit: Callable[[], Awaitable[Any]]) -> Any:
    """The runner's `Perform` under DBOS: `unit` as a checkpointed step.

    Its result goes through JSON on the first run too, so a replay sees the same value.
    """

    async def checkpointed() -> Any:
        async with runtime.fresh_connections():
            return _dump(await unit())

    return _load(await DBOS.run_step_async({"name": label}, checkpointed))


def _dump(value: Any) -> Any:
    if isinstance(value, StepOutcome):
        return {"outcome": {**asdict(value), "output": serialize(value.output)}}
    return {"value": serialize(value)}


def _load(value: Any) -> Any:
    if "outcome" in value:
        return StepOutcome(**value["outcome"])
    return value["value"]


__all__ = [
    "QUEUE",
    "RUN_WORKFLOW",
    "runs_under_other_versions",
]
