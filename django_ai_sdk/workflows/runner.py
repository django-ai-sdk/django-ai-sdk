from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.workflows.steps import (
    OnError,
    StepAlreadyRunning,
    StepFailed,
    StepOutcome,
    WorkflowContext,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.workflows.actions import WorkflowAction
    from django_ai_sdk.workflows.steps import Step

logger = logging.getLogger(__name__)


class _StepRunner:
    """Walks the declared steps once, in order."""

    def __init__(self, steps: Sequence[Step], *, actions: Sequence[WorkflowAction] = ()) -> None:
        self.steps = list(steps)
        self.actions = list(actions)

    async def run(
        self,
        *,
        inputs: Mapping[str, Any] | None = None,
        principal: AbstractBaseUser | AnonymousUser | None = None,
        completed: Mapping[str, Any] | None = None,
        workflow: str = "",
        run_id: str = "",
    ) -> dict[str, StepOutcome]:
        """Run each step whose required steps have completed, in declared order."""
        check_pipeline(self.steps)

        outcomes: dict[str, StepOutcome] = {}
        ctx = WorkflowContext(
            inputs=dict(inputs or {}),
            steps=outcomes,
            principal=principal,
            workflow=workflow,
            run_id=run_id,
        )

        await self._notify(self.actions, "on_run_start", lambda action: action.on_run_start(ctx))
        try:
            await self._walk(ctx, outcomes, dict(completed or {}))
        except Exception as exc:
            await self._end(ctx, exc)
            raise
        await self._end(ctx, None)
        return outcomes

    async def _end(self, ctx: WorkflowContext, error: BaseException | None) -> None:
        await self._notify(self.actions, "on_run_end", lambda action: action.on_run_end(ctx, error))

    @staticmethod
    async def _notify(
        actions: Sequence[WorkflowAction],
        what: str,
        call: Callable[[WorkflowAction], Awaitable[None]],
    ) -> None:
        """Tell each action, and carry on when one of them is broken.

        An action watches the run; it does not do the run's work, so one that raises is
        logged and the walk carries on. `StepAlreadyRunning` is the exception: an action
        whose rows are also a claim refuses a duplicate delivery that way.
        """
        for action in actions:
            try:
                await call(action)
            except StepAlreadyRunning:
                raise
            except Exception:
                logger.exception("Workflow action %s failed in %s", type(action).__name__, what)

    async def _walk(
        self,
        ctx: WorkflowContext,
        outcomes: dict[str, StepOutcome],
        recorded: Mapping[str, Any],
    ) -> None:
        """Run each step in turn, recording every one that will not run."""
        for index, step in enumerate(self.steps):
            if step.name in recorded:
                # Replayed without an action call: the row belongs to the run that
                # completed it, so closing it again would re-date it.
                outcomes[step.name] = StepOutcome(status="completed", output=recorded[step.name])
                continue

            # Completion, not a truthy value: a step that produced None still ran.
            missing = [
                name
                for name in step.requires
                if (done := outcomes.get(name)) is None or done.status != "completed"
            ]
            if missing:
                await self._settle(
                    ctx,
                    step,
                    outcomes,
                    StepOutcome(status="skipped", detail=f"{missing[0]} produced nothing"),
                )
                continue

            if reason := await step.skip_when(ctx):
                await self._settle(
                    ctx, step, outcomes, StepOutcome(status="skipped", detail=reason)
                )
                continue

            await self._notify(
                self._actions_for(step),
                "on_step_start",
                lambda action: action.on_step_start(ctx, step),
            )
            try:
                outcome = await step.run(ctx)
            except Exception as exc:
                logger.warning("Step %s failed: %s", step.name, exc)
                await self._settle(
                    ctx, step, outcomes, StepOutcome(status="failed", detail=str(exc))
                )
                if step.on_error is OnError.FAIL:
                    await self._abandon(ctx, index, outcomes, recorded, f"{step.name} failed")
                    raise
                continue

            await self._settle(ctx, step, outcomes, outcome)
            if outcome.status == "failed" and step.on_error is OnError.FAIL:
                await self._abandon(ctx, index, outcomes, recorded, f"{step.name} failed")
                raise StepFailed(outcome.detail or f"Step {step.name!r} failed.")

    def _actions_for(self, step: Step) -> list[WorkflowAction]:
        """Run-wide actions fire for every step; a step's own fire for it alone."""
        return [*self.actions, *step.actions]

    async def _settle(
        self,
        ctx: WorkflowContext,
        step: Step,
        outcomes: dict[str, StepOutcome],
        outcome: StepOutcome,
    ) -> None:
        """Put the outcome on the run's state and tell the actions."""
        outcomes[step.name] = outcome
        await self._notify(
            self._actions_for(step),
            "on_step_end",
            lambda action: action.on_step_end(ctx, step, outcome),
        )

    async def _abandon(
        self,
        ctx: WorkflowContext,
        index: int,
        outcomes: dict[str, StepOutcome],
        recorded: Mapping[str, Any],
        reason: str,
    ) -> None:
        """Record every step after `index` as skipped, so none is merely absent."""
        for step in self.steps[index + 1 :]:
            if step.name in recorded:
                continue
            await self._settle(ctx, step, outcomes, StepOutcome(status="skipped", detail=reason))


def check_pipeline(steps: Sequence[Any], *, label: str = "A workflow") -> None:
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
    principal: AbstractBaseUser | AnonymousUser | None = None,
    actions: Sequence[WorkflowAction] = (),
    completed: Mapping[str, Any] | None = None,
    workflow: str = "",
    run_id: str = "",
) -> dict[str, StepOutcome]:
    """Run `steps` in declared order, reporting each outcome to `actions`."""
    runner = _StepRunner(steps, actions=actions)
    return await runner.run(
        inputs=inputs,
        principal=principal,
        completed=completed,
        workflow=workflow,
        run_id=run_id,
    )


__all__ = ["check_pipeline", "run_steps"]
