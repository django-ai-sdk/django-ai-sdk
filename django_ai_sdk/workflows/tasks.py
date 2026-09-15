from __future__ import annotations

from asgiref.sync import async_to_sync
from django_tasks import task

from django_ai_sdk.workflows.models import WorkflowRun
from django_ai_sdk.workflows.schemas import WorkflowDefinition


@task(queue_name="default")
def execute_workflow(run_id: str) -> None:
    """Sync task entry point — worker calls this, bridges to async executor."""
    async_to_sync(_execute_async)(run_id)


async def _execute_async(run_id: str) -> None:
    from django_ai_sdk.workflows.executor import WorkflowExecutor  # lazy — breaks circular

    run = await WorkflowRun.objects.aget(id=run_id)
    workflow = WorkflowDefinition.model_validate(run.workflow_definition)
    await WorkflowExecutor().run(workflow, inputs=run.inputs, workflow_run=run)
