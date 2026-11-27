from __future__ import annotations

from typing import TYPE_CHECKING

from django_ai_sdk.workflows.steps import AgentStep, Step, StepOutcome
from pydantic import BaseModel

if TYPE_CHECKING:
    from django_ai_sdk.workflows.steps import WorkflowContext

from .extraction import PirateExtractionAgent


class ThreadDigest(BaseModel):
    title: str
    summary: str


class GatherStep(Step):
    async def run(self, ctx: WorkflowContext) -> StepOutcome:
        from django_ai_sdk.conversation.models import Message

        rows = [
            row
            async for row in Message.objects.filter(thread_id=ctx.input("thread"), is_deleted=False)
            .order_by("created_at")
            .values_list("result", flat=True)
        ]
        text = "\n\n".join(f"{row.get('role', '')}: {row.get('content', '')}" for row in rows)
        if not text.strip():
            return StepOutcome(status="failed", detail="the thread has no messages")
        return StepOutcome(output=text, detail=f"{len(rows)} messages")


class DigestStep(AgentStep):
    agent = PirateExtractionAgent
    schema = ThreadDigest
    instructions = (
        "You summarise conversations. Reply with a title of at most six "
        "words and a two-sentence summary."
    )

    async def user_message(self, ctx: WorkflowContext) -> str:
        return ctx.step("transcript", "")

    async def run(self, ctx: WorkflowContext) -> StepOutcome:
        from django_ai_sdk.conversation.models import Thread

        outcome = await super().run(ctx)
        if outcome.status != "completed":
            return outcome

        title = outcome.output["title"]
        await Thread.objects.filter(id=ctx.input("thread")).aupdate(title=title)
        return StepOutcome(output=outcome.output, detail=title)
