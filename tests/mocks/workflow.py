"""Workflow step and sink fakes.

A step that appends to a list is the whole test double: the model call is
the only boundary these tests care about, and none of them cross it.
"""

from __future__ import annotations

from typing import Any

from django_ai_sdk.workflows import OnError, Step, StepContext, StepOutcome


class FakeStep(Step):
    """A Step whose behaviour is configured per instance.

    `journal`, when given, receives the step's name in `run`, so a test can
    read which steps ran, and in what order, without touching the database.
    """

    def __init__(
        self,
        name: str,
        *,
        provides: str = "",
        requires: tuple[str, ...] = (),
        outcome: StepOutcome | None = None,
        raises: Exception | None = None,
        skip_reason: str = "",
        on_error: OnError = OnError.FAIL,
        error_key: str = "",
        journal: list[str] | None = None,
    ) -> None:
        self.name = name
        self.provides = provides
        self.requires = tuple(requires)
        self.outcome = outcome if outcome is not None else StepOutcome(detail=name)
        self.raises = raises
        self.skip_reason = skip_reason
        self.on_error = on_error
        self.error_key = error_key
        self.journal = journal
        self.calls = 0

    async def skip_when(self, ctx: StepContext) -> str:
        return self.skip_reason

    async def run(self, ctx: StepContext) -> StepOutcome:
        self.calls += 1
        if self.journal is not None:
            self.journal.append(self.name)
        if self.raises is not None:
            raise self.raises
        return self.outcome


class RecordingSink:
    """A StepSink that appends every call, for assertions."""

    def __init__(self, completed_names: dict[str, Any] | None = None) -> None:
        self.events: list[tuple[str, str, Any]] = []
        self.completed_names: dict[str, Any] = completed_names or {}

    async def completed(self) -> dict[str, Any]:
        return dict(self.completed_names)

    async def begin(self) -> None:
        self.events.append(("begin", "", None))

    async def end(self, outcomes: Any, error: BaseException | None) -> None:
        self.events.append(("end", "", None if error is None else str(error)))

    async def open(self, step: str) -> None:
        self.events.append(("open", step, None))

    async def close(self, step: str, outcome: StepOutcome) -> None:
        self.events.append(("close", step, outcome.status))

    async def fail(self, step: str, exc: BaseException) -> None:
        self.events.append(("fail", step, str(exc)))
