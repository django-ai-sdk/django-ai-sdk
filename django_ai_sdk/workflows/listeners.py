from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django_ai_sdk.workflows.conf import registered

if TYPE_CHECKING:
    from django_ai_sdk.workflows.steps import Step, StepOutcome, WorkflowContext


class RunListener:
    """Receives a run's progress.

    Every callback is a no-op by default, so override only the ones you need. A
    listener on one step gets only that step's callbacks. Run start and end go to
    listeners on the workflow.
    """

    description: str = ""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """`config` comes from the ListenerDefinition. Code may pass nothing."""
        self.config = config or {}

    async def on_run_start(self, ctx: WorkflowContext) -> None:
        """Before the first step is considered."""

    async def on_step_start(self, ctx: WorkflowContext, step: Step) -> None:
        """A step is about to run. Not called for a step that is skipped."""

    async def on_step_end(self, ctx: WorkflowContext, step: Step, outcome: StepOutcome) -> None:
        """A step completed, failed or was skipped."""

    async def on_run_end(self, ctx: WorkflowContext, error: BaseException | None) -> None:
        """The run is over. `error` is what ended it, or None."""


def get_listener_registry() -> dict[str, type[RunListener]]:
    """Listeners a definition may name, by key, from `AI_SDK_WORKFLOWS['LISTENERS']`.

    Read at call time, so a settings change needs no restart. A path that does not
    import is left out with a warning (and system check E003).
    """
    return registered("LISTENERS", RunListener)


__all__ = ["RunListener", "get_listener_registry"]
