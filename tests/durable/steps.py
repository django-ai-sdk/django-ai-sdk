"""Steps and a task for the DBOS tests. Each records what it did in `calls`."""

from __future__ import annotations

import asyncio
import threading

from django_tasks import task

from django_ai_sdk.oversight.guard import note_opened
from django_ai_sdk.workflows.steps import Step, StepOutcome, WorkflowContext

# Step name -> how many times it ran, across the worker's threads.
calls: dict[str, int] = {}


class EchoStep(Step):
    """Completes with the run's `text` input and the step's name."""

    async def run(self, ctx: WorkflowContext) -> StepOutcome:
        calls[self.name] = calls.get(self.name, 0) + 1
        return StepOutcome(output=f"{self.name}:{ctx.input('text', '')}")


class TurnStep(Step):
    """Completes as if its model call opened the oversight turn "turn-1"."""

    async def run(self, ctx: WorkflowContext) -> StepOutcome:
        class Guard:
            id = "turn-1"

        note_opened(Guard())
        return StepOutcome(output="turned")


@task(queue_name="default")
def add(a: int, b: int) -> int:
    return a + b


# Set to let HoldStep finish; cleared, it waits. A worker "dies" while it waits.
release = threading.Event()


class HoldStep(Step):
    """Completes once `release` is set, so a test can stop the worker mid-step."""

    async def run(self, ctx: WorkflowContext) -> StepOutcome:
        calls[self.name] = calls.get(self.name, 0) + 1
        while not release.is_set():
            await asyncio.sleep(0.05)
        return StepOutcome(output="held")
