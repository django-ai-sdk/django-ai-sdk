"""Recording a workflow run's step-by-step progress.

A sink is built by the host and passed to `run_steps`, so each workflow records
against whatever table it owns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from django.utils import timezone

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from django_ai_sdk.workflows.models import WorkflowRun
    from django_ai_sdk.workflows.steps import Step, StepOutcome


class StepSink(Protocol):
    """What the runner reports a run's progress to."""

    async def completed(self) -> Mapping[str, Any]:
        """Steps already finished, name to stored output, for resume."""
        ...

    async def begin(self) -> None:
        """Called before the first step is considered."""
        ...

    async def end(self, outcomes: Mapping[str, StepOutcome], error: BaseException | None) -> None:
        """Called when the run is over. `error` is what ended it, or None."""
        ...

    async def open(self, step: str) -> None: ...

    async def close(self, step: str, outcome: StepOutcome) -> None: ...

    async def fail(self, step: str, exc: BaseException) -> None: ...


class WorkflowRunStepSink:
    """Records one WorkflowRunStep row per step of one run.

    `steps` is the pipeline as declared, which fixes each row's `sequence`.
    """

    def __init__(self, run: WorkflowRun, steps: Sequence[Step]) -> None:
        self.run = run
        self.sequence = {step.name: index for index, step in enumerate(steps)}
        self.output_keys = {step.name: step.provides or step.name for step in steps}

    async def begin(self) -> None:
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

    async def end(self, outcomes: Mapping[str, StepOutcome], error: BaseException | None) -> None:
        """Stamp failure on the run when the walk aborted; success is the executor's."""
        from django_ai_sdk.workflows.models import WorkflowRun

        if error is None:
            return
        self.run.status = WorkflowRun.Status.FAILED
        self.run.error = str(error)
        self.run.completed_at = timezone.now()
        await self.run.asave(update_fields=["status", "error", "completed_at", "updated_at"])

    async def completed(self) -> Mapping[str, Any]:
        """Stored output of every step that already finished."""
        from django_ai_sdk.workflows.models import WorkflowRunStep

        return {
            row.step_name: row.output
            async for row in self.run.steps.filter(status=WorkflowRunStep.Status.COMPLETED)
        }

    async def open(self, step: str) -> None:
        """Mark a step running, clearing anything a previous attempt left."""
        from django_ai_sdk.workflows.models import WorkflowRunStep

        await WorkflowRunStep.objects.aupdate_or_create(
            run=self.run,
            sequence=self.sequence[step],
            defaults={
                "step_name": step,
                "output_key": self.output_keys[step],
                "status": WorkflowRunStep.Status.RUNNING,
                "started_at": timezone.now(),
                "output": None,
                "error": "",
                "detail": "",
            },
        )

    async def close(self, step: str, outcome: StepOutcome) -> None:
        """Write a step's outcome. Upserts, because a skipped step never opened."""
        from django_ai_sdk.workflows.models import WorkflowRunStep

        await WorkflowRunStep.objects.aupdate_or_create(
            run=self.run,
            sequence=self.sequence[step],
            defaults={
                "step_name": step,
                "output_key": self.output_keys[step],
                "status": WorkflowRunStep.Status(outcome.status),
                "output": _serialize(outcome.output),
                "error": outcome.detail if outcome.status == "failed" else "",
                "detail": outcome.detail[:255],
                "completed_at": timezone.now(),
            },
        )

    async def fail(self, step: str, exc: BaseException) -> None:
        """Mark a step failed after it raised."""
        from django_ai_sdk.workflows.models import WorkflowRunStep

        await self.run.steps.filter(sequence=self.sequence[step]).aupdate(
            status=WorkflowRunStep.Status.FAILED,
            error=str(exc),
            detail=str(exc)[:255],
            completed_at=timezone.now(),
        )


def _serialize(output: Any) -> Any:
    """JSON-safe form of a step's output."""
    if output is None or isinstance(output, str | int | float | bool | list | dict):
        return output
    dump = getattr(output, "model_dump", None)
    return dump(mode="json") if callable(dump) else str(output)


__all__ = ["StepSink", "WorkflowRunStepSink"]
