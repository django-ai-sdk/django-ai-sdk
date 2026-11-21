from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.oversight.calls import requires_turn
from django_ai_sdk.oversight.guard import watching_turns
from django_ai_sdk.workflows.steps import (
    OnError,
    RunSuperseded,
    StepOutcome,
    StepRecorded,
    WorkflowContext,
    WorkflowStopped,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.workflows.listeners import RunListener
    from django_ai_sdk.workflows.steps import Step

logger = logging.getLogger(__name__)


# How the walk hands over a unit of work: a label, and a coroutine function that
# does the unit's side effects and returns its result. In-process it is awaited
# as is. Under DBOS (`workflows.durable`) each unit is a checkpointed step, so a
# recovered run replays a finished unit's result instead of doing it again.
Perform = Callable[[str, Callable[[], Awaitable[Any]]], Awaitable[Any]]
# How a waiting step gets its decision: a topic and a timeout in seconds. None
# when the run cannot wait, which only a durable run can.
Wait = Callable[[str, float], Awaitable[Any]]

# How long a step whose wait ran out waits again for a decision taken just then.
DECISION_IN_FLIGHT_SECONDS = 60.0


async def perform_now(label: str, unit: Callable[[], Awaitable[Any]]) -> Any:
    """Do `unit` here and now: the in-process `Perform`."""
    return await unit()


class _StepRunner:
    """Walks the declared steps once, in order.

    The walk itself only decides. Every side effect (running a step, telling the
    recorder and listeners) happens inside a unit given to `perform`, so the walk
    is the same whether it runs in-process or as a DBOS workflow.
    """

    def __init__(
        self,
        steps: Sequence[Step],
        *,
        listeners: Sequence[RunListener],
        recorder: RunListener | None,
        completed: Mapping[str, Any],
        perform: Perform = perform_now,
        wait: Wait | None = None,
        **context: Any,
    ) -> None:
        self.steps = list(steps)
        self.outcomes: dict[str, StepOutcome] = {}
        self.ctx = WorkflowContext(
            steps=self.outcomes, declared={step.name: step for step in self.steps}, **context
        )
        self.listeners = list(listeners)
        self.recorder = recorder
        self.recorded = completed
        self.perform = perform
        self.wait = wait
        # The exception the last step raised, so an in-process run raises it as is.
        self.raised: Exception | None = None

    async def run(self) -> dict[str, StepOutcome]:
        """Run each step whose required steps have completed, in declared order."""
        ctx = self.ctx
        await self.perform(
            "run:start",
            lambda: self._notify(self.listeners, lambda listener: listener.on_run_start(ctx)),
        )
        try:
            await self._walk()
        except Exception as exc:
            await self.perform("run:end", lambda error=exc: self._end(error))
            raise
        await self.perform("run:end", lambda: self._end(None))
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
        for index, step in enumerate(self.steps):
            if step.name in self.recorded:
                # Replayed without a listener call: the row belongs to the run that
                # completed it, so closing it again would re-date it.
                self.outcomes[step.name] = StepOutcome(
                    status="completed", output=self.recorded[step.name]
                )
                continue

            if step.waits and self.wait is not None:
                outcome = await self._decided(step)
            else:
                outcome = await self.perform(f"step:{step.name}", lambda s=step: self._attempt(s))
            self.outcomes[step.name] = outcome

            if outcome.status == "failed" and step.on_error is OnError.STOP:
                await self.perform(
                    f"abandon:{step.name}", lambda i=index: self._abandon(i, f"{step.name} failed")
                )
                raised, self.raised = self.raised, None
                raise raised or WorkflowStopped(outcome.detail or f"Step {step.name!r} failed.")
            self.raised = None

    async def _attempt(self, step: Step, *, waiting: bool = False) -> StepOutcome:
        """One step: skip it, run it, or (`waiting`) set it waiting. Records the outcome."""
        ctx = self.ctx
        # Completion, not a truthy value: a step that produced None still ran.
        missing = [
            name
            for name in step.requires
            if (done := self.outcomes.get(name)) is None or done.status != "completed"
        ]
        if missing:
            waited = self.outcomes.get(missing[0])
            how = "failed" if waited and waited.status == "failed" else "was skipped"
            return await self._settle(
                step, StepOutcome(status="skipped", detail=f"{missing[0]} {how}")
            )

        if reason := await step.skip_if(ctx):
            return await self._settle(step, StepOutcome(status="skipped", detail=reason))

        try:
            await self._notify(
                self._listeners_for(step), lambda listener: listener.on_step_start(ctx, step)
            )
        except StepRecorded as recorded:
            self.outcomes[step.name] = recorded.outcome
            return recorded.outcome
        if waiting:
            # Only the recorder hears it: for the listeners the step has not ended.
            return await self._settle(
                step, StepOutcome(status="waiting", detail=step.waiting_for()), listeners=False
            )

        opened: list[str] = []
        try:
            # A step's model calls must come from a turn.
            with watching_turns() as opened, requires_turn(step.channel(ctx)):
                outcome = await step.run(ctx)
        except Exception as exc:
            logger.warning("Step %s failed: %s", step.name, exc)
            self.raised = exc
            outcome = StepOutcome(status="failed", detail=str(exc))
        if opened and not outcome.turn_id:
            outcome.turn_id = opened[0]
        return await self._settle(step, outcome)

    async def _decided(self, step: Step) -> StepOutcome:
        """A waiting step: set it waiting, wait for the decision, then record it."""
        pending = await self.perform(f"step:{step.name}", lambda: self._attempt(step, waiting=True))
        if pending.status != "waiting":
            return pending
        assert self.wait is not None
        seconds = step.wait_seconds()
        while (decision := await self.wait(f"approval:{step.name}", seconds)) is None:
            # Out of time, unless a decision was taken meanwhile and is on its way.
            if await self.perform(f"expire:{step.name}", lambda: self._expire(step)):
                break
            seconds = DECISION_IN_FLIGHT_SECONDS
        return await self.perform(
            f"decide:{step.name}", lambda: self._settle(step, step.decide(decision))
        )

    async def _expire(self, step: Step) -> bool:
        """End `step`'s wait. False when a decision was taken first: it is on its way."""
        expire = getattr(self.recorder, "expire", None)
        return True if expire is None else await expire(step)

    def _listeners_for(self, step: Step) -> list[RunListener]:
        """Run-wide listeners fire for every step; a step's own fire for it alone."""
        return [*self.listeners, *step.listeners]

    async def _settle(
        self, step: Step, outcome: StepOutcome, *, listeners: bool = True
    ) -> StepOutcome:
        """Put the outcome on the run's state and tell the recorder and listeners."""
        self.outcomes[step.name] = outcome
        await self._notify(
            self._listeners_for(step) if listeners else [],
            lambda listener: listener.on_step_end(self.ctx, step, outcome),
        )
        return outcome

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
    perform: Perform = perform_now,
    wait: Wait | None = None,
) -> dict[str, StepOutcome]:
    """Run `steps` in declared order, reporting each outcome to `listeners`.

    `recorder` (a `RunRecorder`) is called first. Its errors stop the run, while a
    listener's errors are only logged. `perform` and `wait` are how a durable run
    checkpoints each unit and waits for a decision (`workflows.durable`).
    """
    validate_steps(steps)
    return await _StepRunner(
        steps,
        listeners=listeners,
        recorder=recorder,
        completed=dict(completed or {}),
        perform=perform,
        wait=wait,
        inputs=dict(inputs or {}),
        user=user,
        workflow=workflow,
        run_id=run_id,
    ).run()


__all__ = ["Perform", "Wait", "perform_now", "run_steps", "validate_steps"]
