"""Workflow step and hook fakes.

A step that appends to a list is the whole test double: the model call is
the only boundary these tests care about, and none of them cross it.
"""

from __future__ import annotations

from typing import Any

from django_ai_sdk.workflows import OnError, Step, StepOutcome, WorkflowContext, WorkflowHook


class FakeStep(Step):
    """A Step whose behaviour is configured per instance.

    `journal`, when given, receives the step's name in `run`, so a test can
    read which steps ran, and in what order, without touching the database.
    """

    def __init__(
        self,
        name: str,
        *,
        requires: tuple[str, ...] = (),
        outcome: StepOutcome | None = None,
        raises: Exception | None = None,
        skip_reason: str = "",
        on_error: OnError = OnError.FAIL,
        hooks: tuple[WorkflowHook, ...] = (),
        journal: list[str] | None = None,
    ) -> None:
        self.name = name
        self.requires = tuple(requires)
        self.outcome = outcome if outcome is not None else StepOutcome(detail=name)
        self.raises = raises
        self.skip_reason = skip_reason
        self.on_error = on_error
        self.hooks = tuple(hooks)
        self.journal = journal
        self.calls = 0

    async def skip_when(self, ctx: WorkflowContext) -> str:
        return self.skip_reason

    async def run(self, ctx: WorkflowContext) -> StepOutcome:
        self.calls += 1
        if self.journal is not None:
            self.journal.append(self.name)
        if self.raises is not None:
            raise self.raises
        return self.outcome


class RecordingHook(WorkflowHook):
    """A WorkflowHook that appends every call, for assertions."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.events: list[tuple[str, str, Any]] = []

    async def on_run_start(self, ctx: WorkflowContext) -> None:
        self.events.append(("run_start", "", None))

    async def on_run_end(self, ctx: WorkflowContext, error: BaseException | None) -> None:
        self.events.append(("run_end", "", None if error is None else str(error)))

    async def on_step_start(self, ctx: WorkflowContext, step: Step) -> None:
        self.events.append(("step_start", step.name, None))

    async def on_step_end(self, ctx: WorkflowContext, step: Step, outcome: StepOutcome) -> None:
        self.events.append(("step_end", step.name, outcome.status))
