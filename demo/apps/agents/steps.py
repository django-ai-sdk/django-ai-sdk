"""Registered workflow step types for the demo.

Both are named by `AI_SDK_WORKFLOW_STEPS` and composed by the `thread-digest`
definition in workflows.py. The package ships no step types; these are the host's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from django_ai_sdk.workflows import AgentStep, Step, StepOutcome
from pydantic import BaseModel

if TYPE_CHECKING:
    from django_ai_sdk.workflows import StepContext

from .extraction import PirateExtractionAgent


class ThreadDigest(BaseModel):
    title: str
    summary: str


class GatherStep(Step):
    """The non-agent step: plain Python over the thread's own rows."""

    async def run(self, ctx: StepContext) -> StepOutcome:
        from django_ai_sdk.conversation.models import Message

        # A run's inputs cross the queue as JSON, so the subject arrives as an id
        # and the step resolves it.
        rows = [
            row
            async for row in Message.objects.filter(thread_id=ctx.get("thread"), is_deleted=False)
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

    async def system_prompt(self, ctx: StepContext) -> str:
        return (
            "You summarise conversations. Reply with a title of at most six "
            "words and a two-sentence summary."
        )

    async def user_message(self, ctx: StepContext) -> str:
        return ctx.get("transcript")

    async def run(self, ctx: StepContext) -> StepOutcome:
        from django_ai_sdk.conversation.models import Thread

        outcome = await super().run(ctx)
        if outcome.status != "completed":
            return outcome

        # Writing the title is this host's business, not the runner's: the step
        # returns the digest, and does what its own product needs.
        await Thread.objects.filter(id=ctx.get("thread")).aupdate(title=outcome.output.title)
        return StepOutcome(output=outcome.output.model_dump(), detail=outcome.output.title)
