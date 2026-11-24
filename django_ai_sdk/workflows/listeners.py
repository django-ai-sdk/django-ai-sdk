from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django_ai_sdk.workflows.conf import registered

if TYPE_CHECKING:
    from django_ai_sdk.workflows.steps import Step, StepOutcome, WorkflowContext


class RunListener:
    """Receives a run's progress; callbacks default to no-ops.

    Step listeners get only that step's callbacks; run start/end go to workflow listeners.
    """

    description: str = ""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """`config` comes from the ListenerDefinition. Code may pass nothing."""
        self.config = config or {}

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> None:
        """Raise ValueError unless this listener accepts `config` (default: none)."""
        if config:
            raise ValueError(f"{cls.__name__} takes no config, got keys {sorted(config)}.")

    async def on_run_start(self, ctx: WorkflowContext) -> None:
        """Before the first step is considered."""

    async def on_step_start(self, ctx: WorkflowContext, step: Step) -> None:
        """A step is about to run. Not called for a step that is skipped."""

    async def on_step_end(self, ctx: WorkflowContext, step: Step, outcome: StepOutcome) -> None:
        """A step completed, failed or was skipped."""

    async def on_run_end(self, ctx: WorkflowContext, error: BaseException | None) -> None:
        """The run is over. `error` is what ended it, or None."""


def get_listener_registry() -> dict[str, type[RunListener]]:
    """Listeners a definition may name, by key, from `AI_SDK_WORKFLOWS['LISTENERS']`."""
    return registered("LISTENERS", RunListener)


__all__ = ["RunListener", "get_listener_registry"]
