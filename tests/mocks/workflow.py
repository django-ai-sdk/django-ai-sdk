from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from django_ai_sdk.workflows.listeners import RunListener
from django_ai_sdk.workflows.schemas import StepDefinition, WorkflowDefinition
from django_ai_sdk.workflows.steps import OnError, Step, StepOutcome, WorkflowContext

AGENT_GET = "django_ai_sdk.agents.services.AgentService.get"
ENQUEUE = "django_ai_sdk.workflows.runs.requeue"


def fake_agent(result: Any = "agent result") -> MagicMock:
    agent = MagicMock()
    agent.run = AsyncMock(return_value=result)
    return agent


def two_steps() -> WorkflowDefinition:
    return WorkflowDefinition(
        steps=[
            StepDefinition(name="first", agent_id="a1"),
            StepDefinition(name="second", agent_id="a1", requires=["first"]),
        ]
    )


class FakeStep(Step):
    def __init__(
        self,
        name: str,
        *,
        requires: tuple[str, ...] = (),
        outcome: StepOutcome | None = None,
        raises: Exception | None = None,
        skip_reason: str = "",
        on_error: OnError = OnError.STOP,
        listeners: tuple[RunListener, ...] = (),
        journal: list[str] | None = None,
    ) -> None:
        self.name = name
        self.requires = tuple(requires)
        self.outcome = outcome if outcome is not None else StepOutcome(detail=name)
        self.raises = raises
        self.skip_reason = skip_reason
        self.on_error = on_error
        self.listeners = tuple(listeners)
        self.journal = journal
        self.calls = 0

    async def skip_if(self, ctx: WorkflowContext) -> str:
        return self.skip_reason

    async def run(self, ctx: WorkflowContext) -> StepOutcome:
        self.calls += 1
        if self.journal is not None:
            self.journal.append(self.name)
        if self.raises is not None:
            raise self.raises
        return self.outcome


class RecordingListener(RunListener):
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


async def run_now(
    workflow: Any,
    *,
    inputs: dict[str, Any] | None = None,
    user: Any = None,
    run: Any = None,
) -> tuple[dict[str, Any], Any]:
    from django_ai_sdk.workflows.runs import execute_run, open_run

    if run is None:
        run = await open_run(workflow, inputs=inputs, user=user)
    await execute_run(str(run.pk))
    await run.arefresh_from_db()
    return run.outputs or {}, run
