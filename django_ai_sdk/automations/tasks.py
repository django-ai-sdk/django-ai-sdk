"""The worker side: execute one claimed AutomationRun.

The workflow runs inline rather than through WorkflowService.run(); across two queue
hops the run could only record that something was handed off, never its outcome.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from asgiref.sync import async_to_sync
from django.utils import timezone
from django_tasks import task

from django_ai_sdk.automations.models import AutomationRun
from django_ai_sdk.automations.registry import get_automation
from django_ai_sdk.tasks import aget_principal

if TYPE_CHECKING:
    from datetime import datetime

    from django_ai_sdk.automations.base import Automation

logger = logging.getLogger(__name__)


class _MissingWorkflow(Exception):
    """The named workflow is not registered."""

    def __init__(self, name: str) -> None:
        super().__init__(
            f"workflow {name!r} is not registered; declare it in an app's workflows.py"
        )


@task(queue_name="default")
def execute_automation(run_id: str) -> dict[str, Any] | None:
    """Sync task entry point — the worker calls this, it bridges to async."""
    return async_to_sync(run_automation)(run_id)


async def run_automation(run_id: str) -> dict[str, Any] | None:
    """Execute one run, recording a terminal status on every exit path."""
    try:
        run = await AutomationRun.objects.select_related("state").aget(id=run_id)
    except AutomationRun.DoesNotExist:
        # The run was deleted between enqueue and pickup.
        return None

    # _finish releases the lease, so the cleanup below must know whether it ran.
    finished = False

    async def finish(**kwargs: Any) -> None:
        nonlocal finished
        await _finish(run, **kwargs)
        finished = True

    automation = get_automation(run.name)
    if automation is None:
        # The declaration is gone but the queued task survived a deploy.
        await finish(status=AutomationRun.Status.SKIPPED, skip_reason="no longer declared")
        return None

    await AutomationRun.objects.filter(id=run.id).aupdate(
        status=AutomationRun.Status.RUNNING, started_at=timezone.now()
    )

    try:
        blocked = await _blocked_by_integrations(automation)
        if blocked:
            await finish(status=AutomationRun.Status.SKIPPED, skip_reason=blocked)
            return None

        user = await aget_principal(run.user_id, source=f"Automation run {run.id}")
        outputs = await asyncio.wait_for(
            _run_workflow(automation, run, user), timeout=automation.get_timeout()
        )
        await finish(status=AutomationRun.Status.SUCCEEDED, output=outputs)
        return outputs

    except _MissingWorkflow as exc:
        # The automation is not broken, its dependency is absent.
        await finish(status=AutomationRun.Status.SKIPPED, skip_reason=str(exc))
        return None
    except TimeoutError as exc:
        # asyncio.wait_for's own TimeoutError carries no message.
        await finish(
            status=AutomationRun.Status.FAILED,
            error=str(exc) or f"Timed out after {automation.get_timeout()} seconds",
        )
        raise
    except Exception as exc:
        await finish(status=AutomationRun.Status.FAILED, error=str(exc))
        # Re-raise so django-tasks marks its own result FAILED too.
        raise
    finally:
        # A cancelled task leaves its row RUNNING, so only the lease is dropped here.
        if run.state and not finished:
            await _release_if_dispatch_is_done(run)


async def _run_workflow(automation: Automation, run: AutomationRun, user: Any) -> dict[str, Any]:
    """Resolve this automation's workflow and execute it, linked to the run."""
    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.workflows.executor import WorkflowExecutor
    from django_ai_sdk.workflows.models import WorkflowRun
    from django_ai_sdk.workflows.registry import aget_workflow

    last_run_at = run.state.last_success_at if run.state else None
    definition = await aget_workflow(automation.workflow)
    if definition is None:
        raise _MissingWorkflow(automation.workflow)

    # There is no human turn, so the automation supplies the one the workflow starts from.
    messages = [
        ChatMessage(
            role="user", content=automation.render_input(user=user, last_run_at=last_run_at)
        )
    ]

    workflow_run = await WorkflowRun.objects.acreate(
        workflow=None,
        workflow_definition=definition.model_dump(),
        status=WorkflowRun.Status.PENDING,
        input_messages=[m.model_dump() for m in messages],
        user_id=getattr(user, "pk", None),
    )
    await AutomationRun.objects.filter(id=run.id).aupdate(workflow_run=workflow_run)

    outputs, _ = await WorkflowExecutor().run(
        definition, messages, user=user, workflow_run=workflow_run
    )
    return outputs


async def _blocked_by_integrations(automation: Automation) -> str:
    """A reason string when a required integration isn't usable, else "".

    A degraded upstream makes the run SKIPPED, not FAILED.
    """
    if not automation.requires:
        return ""

    from django_ai_sdk.integrations.base import IntegrationStatus
    from django_ai_sdk.integrations.registry import get_integrations

    integrations = await get_integrations(list(automation.requires))
    missing = set(automation.requires) - set(integrations)
    if missing:
        return f"requires {', '.join(sorted(missing))}, which is not installed"

    for name in automation.requires:
        try:
            status = await integrations[name].get_status()
        except Exception:
            logger.warning("Could not read status for integration %r", name, exc_info=True)
            return f"{name} status could not be read"
        if status != IntegrationStatus.ACTIVE:
            return f"{name} is {status}"
    return ""


async def _finish(
    run: AutomationRun,
    *,
    status: str,
    output: Any = None,
    error: str = "",
    skip_reason: str = "",
) -> None:
    """Write the terminal state, then release the lease if the dispatch is finished."""
    finished_at = timezone.now()
    await AutomationRun.objects.filter(id=run.id).aupdate(
        status=status,
        output=output if isinstance(output, (dict, list, str, int, float, bool)) else None,
        error=error[:10000],
        skip_reason=skip_reason[:255],
        finished_at=finished_at,
    )
    if run.state:
        await _release_if_dispatch_is_done(run, finished_at=finished_at)


async def _release_if_dispatch_is_done(
    run: AutomationRun, *, finished_at: datetime | None = None
) -> None:
    """Drop the lease once no sibling from the same tick is still in flight.

    One claim fans out to one run per principal, so the first finisher freeing it would
    break allow_overlap=False.
    """
    from django_ai_sdk.automations.runner import release

    in_flight = AutomationRun.objects.filter(
        dispatch_id=run.dispatch_id,
        status__in=(AutomationRun.Status.PENDING, AutomationRun.Status.RUNNING),
    ).exclude(id=run.id)
    if await in_flight.aexists():
        return

    # last_success_at is per-automation while the audience is per-user, so any success
    # in the dispatch advances the window.
    succeeded = await AutomationRun.objects.filter(
        dispatch_id=run.dispatch_id, status=AutomationRun.Status.SUCCEEDED
    ).aexists()
    await release(run.state, succeeded_at=(finished_at or timezone.now()) if succeeded else None)


__all__ = ["execute_automation", "run_automation"]
