"""Hooks: everything that watches a run rather than doing its work.

One kind of object, attached in two places. A hook on the `WorkflowDefinition`
sees the run and every step in it; a hook on a `StepDefinition` sees that step alone.
Recording a run's rows, notifying someone a step finished, and delivering the
result when the run ends are all the same shape.

A host declares its own by key in `AI_SDK_WORKFLOW_HOOKS`; the package ships one,
`RunRecorder`, which the executor always attaches.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from django.utils import timezone
from django.utils.module_loading import import_string

from django_ai_sdk.utils import resolve_setting, serialize

if TYPE_CHECKING:
    from collections.abc import Sequence

    from django_ai_sdk.workflows.models import WorkflowRun
    from django_ai_sdk.workflows.steps import Step, StepOutcome, WorkflowContext

logger = logging.getLogger(__name__)


class WorkflowHook:
    """What a run reports its progress to.

    Every callback is a no-op, so a hook implements only the moments it cares about.
    A hook attached to one step gets that step's two callbacks only; the run-level
    pair belongs to hooks attached to the workflow.
    """

    description: str = ""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Built from the `config` its HookDefinition carries, or from nothing in code."""
        self.config = config or {}

    async def on_run_start(self, ctx: WorkflowContext) -> None:
        """Before the first step is considered."""

    async def on_step_start(self, ctx: WorkflowContext, step: Step) -> None:
        """A step is about to run. Not called for a step that is skipped."""

    async def on_step_end(self, ctx: WorkflowContext, step: Step, outcome: StepOutcome) -> None:
        """A step settled — completed, failed or skipped."""

    async def on_run_end(self, ctx: WorkflowContext, error: BaseException | None) -> None:
        """The run is over. `error` is what ended it, or None."""


class RunRecorder(WorkflowHook):
    """Writes one WorkflowRunStep row per step of one run.

    `steps` is the pipeline as declared, which fixes each row's `sequence`.
    """

    description = "Record each step against the WorkflowRun"

    def __init__(self, run: WorkflowRun, steps: Sequence[Step]) -> None:
        # Built by the executor, never named in a definition, so it takes a run
        # rather than a config dict.
        super().__init__()
        self.run = run
        self.sequence = {step.name: index for index, step in enumerate(steps)}

    async def on_run_start(self, ctx: WorkflowContext) -> None:
        """Mark the run row running."""
        from django_ai_sdk.workflows.models import WorkflowRun

        fields: list[str] = []
        if self.run.status != WorkflowRun.Status.RUNNING:
            self.run.status = WorkflowRun.Status.RUNNING
            fields.append("status")
        if not self.run.started_at:
            self.run.started_at = timezone.now()
            fields.append("started_at")
        if fields:
            fields.append("updated_at")
            await self.run.asave(update_fields=fields)

    async def on_run_end(self, ctx: WorkflowContext, error: BaseException | None) -> None:
        """Stamp failure on the run when the walk aborted; success is the executor's."""
        from django_ai_sdk.workflows.models import WorkflowRun
        from django_ai_sdk.workflows.steps import StepAlreadyRunning

        # A refused claim is not this run's failure: the delivery that holds it is
        # still working, and stamping the row here would overwrite its progress.
        if error is None or isinstance(error, StepAlreadyRunning):
            return
        self.run.status = WorkflowRun.Status.FAILED
        self.run.error = str(error)
        self.run.completed_at = timezone.now()
        await self.run.asave(update_fields=["status", "error", "completed_at", "updated_at"])

    async def on_step_start(self, ctx: WorkflowContext, step: Step) -> None:
        """Mark a step running, clearing anything a previous attempt left."""
        from django_ai_sdk.workflows.models import WorkflowRunStep

        await WorkflowRunStep.objects.aupdate_or_create(
            run=self.run,
            sequence=self.sequence[step.name],
            defaults={
                "step_name": step.name,
                "status": WorkflowRunStep.Status.RUNNING,
                "started_at": timezone.now(),
                "output": None,
                "error": "",
                "detail": "",
            },
        )

    async def on_step_end(self, ctx: WorkflowContext, step: Step, outcome: StepOutcome) -> None:
        """Write a step's outcome. Upserts, because a skipped step never started."""
        from django_ai_sdk.workflows.models import WorkflowRunStep

        await WorkflowRunStep.objects.aupdate_or_create(
            run=self.run,
            sequence=self.sequence[step.name],
            defaults={
                "step_name": step.name,
                "status": WorkflowRunStep.Status(outcome.status),
                "output": serialize(outcome.output),
                "error": outcome.detail if outcome.status == "failed" else "",
                "detail": outcome.detail[:255],
                "completed_at": timezone.now(),
            },
        )


def get_hook_registry() -> dict[str, type[WorkflowHook]]:
    """Hooks a definition may name, by key, from `AI_SDK_WORKFLOW_HOOKS`.

    Read at call time, so a settings change needs no restart.
    """
    registry: dict[str, type[WorkflowHook]] = {}
    for key, path in resolve_setting("AI_SDK_WORKFLOW_HOOKS", {}).items():
        try:
            cls = import_string(path)
        except ImportError:
            logger.warning(
                "Workflow hook %r names %r, which does not import. It is not composable.",
                key,
                path,
            )
            continue
        if not isinstance(cls, type) or not issubclass(cls, WorkflowHook):
            logger.warning(
                "Workflow hook %r names %r, which is not a WorkflowHook subclass. "
                "It is not composable.",
                key,
                path,
            )
            continue
        registry[key] = cls
    return registry


__all__ = ["RunRecorder", "WorkflowHook", "get_hook_registry"]
