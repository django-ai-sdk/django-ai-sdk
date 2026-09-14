"""Run a pipeline of Steps in the order they were declared.

The walk is linear, not a topological sort: a step runs once every name it
requires is on the run's state, and a step declared before its producer is
refused rather than reordered.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.workflows.steps import OnError, StepContext, StepFailed, StepOutcome

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.workflows.sink import StepSink
    from django_ai_sdk.workflows.steps import Step

logger = logging.getLogger(__name__)


class StepRunner:
    """Walks the declared steps once, in order."""

    def __init__(self, steps: Sequence[Step], *, sink: StepSink | None = None) -> None:
        self.steps = list(steps)
        self.sink = sink

    async def run(
        self,
        *,
        inputs: Mapping[str, Any] | None = None,
        principal: AbstractBaseUser | AnonymousUser | None = None,
    ) -> dict[str, StepOutcome]:
        """Run each step whose required names are available, in declared order."""
        supplied = dict(inputs or {})
        _validate(self.steps, supplied)

        outputs: dict[str, Any] = {}
        ctx = StepContext(inputs=supplied, outputs=outputs, principal=principal)

        outcomes: dict[str, StepOutcome] = {}
        recorded: Mapping[str, Any] = {}
        if self.sink is not None:
            await self.sink.begin()
            recorded = await self.sink.completed()
        try:
            await self._walk(ctx, recorded, outcomes, outputs, supplied)
        except Exception as exc:
            await self._end(outcomes, exc)
            raise
        await self._end(outcomes, None)
        return outcomes

    async def _end(self, outcomes: dict[str, StepOutcome], error: BaseException | None) -> None:
        if self.sink is not None:
            await self.sink.end(outcomes, error)

    async def _walk(
        self,
        ctx: StepContext,
        recorded: Mapping[str, Any],
        outcomes: dict[str, StepOutcome],
        outputs: dict[str, Any],
        supplied: Mapping[str, Any],
    ) -> None:
        """Run each step in turn, recording every one that will not run."""
        for index, step in enumerate(self.steps):
            if step.name in recorded:
                # Republished without a sink call: the row belongs to the run that
                # completed it, so closing it again would re-date it.
                output = recorded[step.name]
                if step.provides:
                    outputs[step.provides] = output
                outcomes[step.name] = StepOutcome(status="completed", output=output)
                continue

            missing = [
                name for name in step.requires if name not in outputs and name not in supplied
            ]
            if missing:
                await self._settle(
                    step,
                    outcomes,
                    outputs,
                    StepOutcome(status="skipped", detail=f"{missing[0]} was not produced"),
                )
                continue

            if reason := await step.skip_when(ctx):
                await self._settle(
                    step, outcomes, outputs, StepOutcome(status="skipped", detail=reason)
                )
                continue

            if self.sink is not None:
                await self.sink.open(step.name)
            try:
                outcome = await step.run(ctx)
            except Exception as exc:
                # `fail` settles the row, so the step gets no `close`.
                if self.sink is not None:
                    await self.sink.fail(step.name, exc)
                outcomes[step.name] = StepOutcome(status="failed", detail=str(exc))
                logger.warning("Step %s failed: %s", step.name, exc)
                if step.on_error is OnError.FAIL:
                    await self._abandon(index, outcomes, outputs, recorded, f"{step.name} failed")
                    raise
                self._publish_error(step, outputs, detail=str(exc))
                continue

            await self._settle(step, outcomes, outputs, outcome)
            if outcome.status == "failed":
                if step.on_error is OnError.FAIL:
                    await self._abandon(index, outcomes, outputs, recorded, f"{step.name} failed")
                    raise StepFailed(outcome.detail or f"Step {step.name!r} failed.")
                self._publish_error(step, outputs, detail=outcome.detail or "failed")

    @staticmethod
    def _publish_error(step: Step, outputs: dict[str, Any], *, detail: str) -> None:
        """File a failure payload under the step's `error_key`, when it declares one."""
        if not step.error_key:
            return
        outputs[step.error_key] = {"step": step.name, "error": detail}

    async def _settle(
        self,
        step: Step,
        outcomes: dict[str, StepOutcome],
        outputs: dict[str, Any],
        outcome: StepOutcome,
    ) -> None:
        """Record an outcome and put what the step provided on the table."""
        if self.sink is not None:
            await self.sink.close(step.name, outcome)
        outcomes[step.name] = outcome
        if outcome.status == "completed" and step.provides:
            outputs[step.provides] = outcome.output

    async def _abandon(
        self,
        index: int,
        outcomes: dict[str, StepOutcome],
        outputs: dict[str, Any],
        recorded: Mapping[str, Any],
        reason: str,
    ) -> None:
        """Record every step after `index` as skipped, so none is merely absent."""
        for step in self.steps[index + 1 :]:
            if step.name in recorded:
                continue
            await self._settle(
                step, outcomes, outputs, StepOutcome(status="skipped", detail=reason)
            )


def _validate(steps: Sequence[Step], inputs: Mapping[str, Any]) -> None:
    """Raise ImproperlyConfigured unless the pipeline can run as declared."""
    if not steps:
        raise ImproperlyConfigured("A workflow needs at least one step.")

    available = set(inputs)
    producers: dict[str, str] = {}
    seen: set[str] = set()

    for step in steps:
        if not step.name:
            raise ImproperlyConfigured(
                "Every step needs a `name`; its outcome is recorded under it."
            )
        if step.name in seen:
            raise ImproperlyConfigured(f"Duplicate step name {step.name!r}.")
        seen.add(step.name)

        # Checked against what is available *here*, so a step declared ahead of the
        # one producing its input is refused rather than reordered.
        unresolved = sorted(name for name in step.requires if name not in available)
        if unresolved:
            raise ImproperlyConfigured(
                f"Step {step.name!r} requires {unresolved}, which the run's inputs do not carry "
                f"and no earlier step provides. Available by this point: {sorted(available)}."
            )

        for claimed, label in ((step.provides, "provides"), (step.error_key, "error_key")):
            if not claimed:
                continue
            if claimed in inputs:
                raise ImproperlyConfigured(
                    f"Step {step.name!r} {label} {claimed!r}, which the run's inputs already "
                    f"carry. One name, one source: rename the step's output or the input."
                )
            if claimed in producers:
                raise ImproperlyConfigured(
                    f"Steps {producers[claimed]!r} and {step.name!r} both claim {claimed!r}. "
                    f"One name, one producer."
                )
            producers[claimed] = step.name
            available.add(claimed)


async def run_steps(
    steps: Sequence[Step],
    *,
    inputs: Mapping[str, Any] | None = None,
    principal: AbstractBaseUser | AnonymousUser | None = None,
    sink: StepSink | None = None,
) -> dict[str, StepOutcome]:
    """Run `steps` in declared order, recording each outcome to `sink` when given."""
    runner = StepRunner(steps, sink=sink)
    return await runner.run(inputs=inputs, principal=principal)


__all__ = ["StepRunner", "run_steps"]
