from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.oversight.calls import requires_turn
from django_ai_sdk.workflows.steps import (
    OnError,
    RunSuperseded,
    StepOutcome,
    WorkflowContext,
    WorkflowStopped,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.workflows.listeners import RunListener
    from django_ai_sdk.workflows.steps import Step

logger = logging.getLogger(__name__)


class _StepRunner:
    """Walks the declared steps once, in order."""

    def __init__(
        self,
        steps: Sequence[Step],
        *,
        listeners: Sequence[RunListener],
        recorder: RunListener | None,
        completed: Mapping[str, Any],
        **context: Any,
    ) -> None:
        self.steps = list(steps)
        self.outcomes: dict[str, StepOutcome] = {}
        self.ctx = WorkflowContext(steps=self.outcomes, **context)
        self.listeners = list(listeners)
        self.recorder = recorder
        self.recorded = completed

    async def run(self) -> dict[str, StepOutcome]:
        """Run each step whose required steps have completed, in declared order."""
        ctx = self.ctx
        await self._notify(self.listeners, lambda listener: listener.on_run_start(ctx))
        try:
            await self._walk()
        except Exception as exc:
            await self._end(exc)
            raise
        await self._end(None)
        return self.outcomes

    async def _end(self, error: BaseException | None) -> None:
        # A superseded attempt is not the run's end: the attempt that owns it tells them.
        listeners = [] if isinstance(error, RunSuperseded) else self.listeners
        await self._notify(listeners, lambda listener: listener.on_run_end(self.ctx, error))

    async def _notify(
        self, listeners: Sequence[RunListener], call: Callable[[RunListener], Awaitable[None]]
    ) -> None:
        """Call the recorder, then each listener.

        A listener only watches the run, so one that raises is logged and the walk
        goes on. A recorder error stops the run, because a step must not run without
        its row.
        """
        if self.recorder is not None:
            await call(self.recorder)
        for listener in listeners:
            try:
                await call(listener)
            except Exception:
                logger.exception("Workflow listener %s failed", type(listener).__name__)

    async def _walk(self) -> None:
        """Run each step in turn, recording every one that will not run."""
        ctx, outcomes = self.ctx, self.outcomes
        for index, step in enumerate(self.steps):
            if step.name in self.recorded:
                # Replayed without a listener call: the row belongs to the run that
                # completed it, so closing it again would re-date it.
                outcomes[step.name] = StepOutcome(
                    status="completed", output=self.recorded[step.name]
                )
                continue

            # Completion, not a truthy value: a step that produced None still ran.
            missing = [
                name
                for name in step.requires
                if (done := outcomes.get(name)) is None or done.status != "completed"
            ]
            if missing:
                waited = outcomes.get(missing[0])
                how = "failed" if waited and waited.status == "failed" else "was skipped"
                await self._settle(
                    step, StepOutcome(status="skipped", detail=f"{missing[0]} {how}")
                )
                continue

            if reason := await step.skip_if(ctx):
                await self._settle(step, StepOutcome(status="skipped", detail=reason))
                continue

            await self._notify(
                self._listeners_for(step), lambda listener: listener.on_step_start(ctx, step)
            )
            try:
                # Oversight: a step's model calls must come from a turn (`oversight.calls`).
                with requires_turn(step.channel(ctx)):
                    outcome = await step.run(ctx)
            except Exception as exc:
                logger.warning("Step %s failed: %s", step.name, exc)
                await self._settle(step, StepOutcome(status="failed", detail=str(exc)))
                if step.on_error is OnError.STOP:
                    await self._abandon(index, f"{step.name} failed")
                    raise
                continue

            await self._settle(step, outcome)
            if outcome.status == "failed" and step.on_error is OnError.STOP:
                await self._abandon(index, f"{step.name} failed")
                raise WorkflowStopped(outcome.detail or f"Step {step.name!r} failed.")

    def _listeners_for(self, step: Step) -> list[RunListener]:
        """Run-wide listeners fire for every step; a step's own fire for it alone."""
        return [*self.listeners, *step.listeners]

    async def _settle(self, step: Step, outcome: StepOutcome) -> None:
        """Put the outcome on the run's state and tell the listeners."""
        self.outcomes[step.name] = outcome
        await self._notify(
            self._listeners_for(step),
            lambda listener: listener.on_step_end(self.ctx, step, outcome),
        )

    async def _abandon(self, index: int, reason: str) -> None:
        """Record every step after `index` as skipped, so none is merely absent."""
        for step in self.steps[index + 1 :]:
            if step.name not in self.recorded:
                await self._settle(step, StepOutcome(status="skipped", detail=reason))


def validate_steps(steps: Sequence[Any], *, label: str = "A workflow") -> None:
    """Raise ImproperlyConfigured unless the steps can run in the order given.

    Takes declarations or compiled Steps: both carry `name` and `requires`.
    """
    if not steps:
        raise ImproperlyConfigured(f"{label} has no steps to run.")

    seen: set[str] = set()
    for step in steps:
        if not step.name:
            raise ImproperlyConfigured(
                f"{label}: every step needs a `name`; its outcome is recorded under it."
            )
        if step.name in seen:
            raise ImproperlyConfigured(f"{label}: duplicate step name {step.name!r}.")

        if unresolved := sorted(name for name in step.requires if name not in seen):
            raise ImproperlyConfigured(
                f"{label}: step {step.name!r} requires {unresolved}, which no earlier "
                f"step produces. Declared by this point: {sorted(seen)}."
            )
        seen.add(step.name)


async def run_steps(
    steps: Sequence[Step],
    *,
    inputs: Mapping[str, Any] | None = None,
    user: AbstractBaseUser | AnonymousUser | None = None,
    listeners: Sequence[RunListener] = (),
    recorder: RunListener | None = None,
    completed: Mapping[str, Any] | None = None,
    workflow: str = "",
    run_id: str = "",
) -> dict[str, StepOutcome]:
    """Run `steps` in declared order, reporting each outcome to `listeners`.

    `recorder` (a `RunRecorder`) is called first. Its errors stop the run, while a
    listener's errors are only logged.
    """
    validate_steps(steps)
    return await _StepRunner(
        steps,
        listeners=listeners,
        recorder=recorder,
        completed=dict(completed or {}),
        inputs=dict(inputs or {}),
        user=user,
        workflow=workflow,
        run_id=run_id,
    ).run()


__all__ = ["run_steps", "validate_steps"]
