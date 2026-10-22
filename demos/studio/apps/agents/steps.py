"""Registered workflow step types for the demo.

Both are named by `AI_SDK_WORKFLOW_STEPS` and composed by the `thread-digest`
definition in workflows.py. The package ships no step types; these are the host's.

Two namespaces, so nothing collides: `ctx.input(...)` is what the caller supplied,
`ctx.step(...)` is what an earlier step produced.
"""

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
    """The non-agent step: plain Python over the thread's own rows."""

    async def run(self, ctx: WorkflowContext) -> StepOutcome:
        from django_ai_sdk.conversation.models import Message

        # A run's inputs cross the queue as JSON, so the subject arrives as an id
        # and the step resolves it.
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
    """The agent step: condenses the transcript and writes the thread's title."""

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

        # Writing the title is this host's business, not the runner's: the step
        # returns the digest, and does what its own product needs.
        title = outcome.output["title"]
        await Thread.objects.filter(id=ctx.input("thread")).aupdate(title=title)
        return StepOutcome(output=outcome.output, detail=title)
