from __future__ import annotations

import asyncio
import threading

from django_tasks import task

from django_ai_sdk.oversight.guard import note_opened
from django_ai_sdk.workflows.steps import ApprovalStep, Step, StepOutcome, WorkflowContext

calls: dict[str, int] = {}


class EchoStep(Step):
    async def run(self, ctx: WorkflowContext) -> StepOutcome:
        calls[self.name] = calls.get(self.name, 0) + 1
        return StepOutcome(output=f"{self.name}:{ctx.input('text', '')}")


class TurnStep(Step):
    async def run(self, ctx: WorkflowContext) -> StepOutcome:
        class Guard:
            id = "turn-1"

        note_opened(Guard())
        return StepOutcome(output="turned")


@task(queue_name="default")
def add(a: int, b: int) -> int:
    return a + b


release = threading.Event()
keys: list[str] = []


class HoldStep(Step):
    async def run(self, ctx: WorkflowContext) -> StepOutcome:
        calls[self.name] = calls.get(self.name, 0) + 1
        keys.append(self.idempotency_key(ctx))
        while not release.is_set():
            await asyncio.sleep(0.05)
        return StepOutcome(output="held")


class QuickApprovalStep(ApprovalStep):
    def wait_seconds(self) -> float:
        return 1.0
