"""Workflow actions for the demo, named by `AI_SDK_WORKFLOW_ACTIONS`.

The package ships no actions a definition can name — delivering a result somewhere
is the host's business, not the SDK's. These two are this host's.
"""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

from django_ai_sdk.workflows.actions import WorkflowAction

if TYPE_CHECKING:
    from django_ai_sdk.workflows.steps import Step, StepOutcome, WorkflowContext


class ConsoleLogAction(WorkflowAction):
    """Print every step's outcome as it settles (dev/debug).

    On the workflow it prints every step; on one step it prints that one.
    """

    description = "Print step outcomes to the console"

    async def on_step_end(self, ctx: WorkflowContext, step: Step, outcome: StepOutcome) -> None:
        print(  # noqa: T201
            f"[{ctx.workflow or ctx.run_id}] {step.name}: {outcome.status} "
            f"{json.dumps(outcome.output, indent=2, default=str)}"
        )


class ThreadMessageAction(WorkflowAction):
    """Post a step's output into a new chat thread owned by the run's user.

    Config: `agent_id` — whose storage adapter opens the thread — and `step`, the
    step whose output is posted. Costs an extra model call when that agent has
    `title_generation` on, since the thread is retitled from its content.
    """

    description = "Post a step's result into a new chat thread for the run's user"

    async def on_run_end(self, ctx: WorkflowContext, error: BaseException | None) -> None:
        from django_ai_sdk.common import ChatMessage
        from django_ai_sdk.storage.services import ThreadService

        if error is not None:
            return
        agent_id = self.config.get("agent_id", "")
        payload = ctx.step(self.config.get("step", ""))
        if not agent_id or payload is None:
            return
        if ctx.principal is None or ctx.principal.is_anonymous:
            return

        content = _as_text(payload)
        source = ctx.workflow or f"workflow:{ctx.run_id}"
        thread_id = await ThreadService.create_thread(
            agent_id,
            title=source,
            metadata={"created_via": "action:thread_message"},
            user=ctx.principal,
        )
        storage = await ThreadService.storage_for_thread(thread_id, user=ctx.principal)
        await storage.store_chat_message(
            # Adapters take an id rather than minting one.
            ChatMessage(id=str(uuid.uuid4()), role="assistant", content=content)
        )
        await _retitle(thread_id, content, agent_id, ctx)


async def _retitle(thread_id: str, content: str, agent_id: str, ctx: WorkflowContext) -> None:
    """Replace the fallback title with a generated one, when the agent wants titles."""
    from django_ai_sdk.agents.services import AgentService
    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.conversation.utils import generate_thread_title
    from django_ai_sdk.storage.services import ThreadService

    try:
        agent = await AgentService.get(agent_id)
    except ValueError:
        return
    if not agent.title_generation:
        return

    title = await generate_thread_title(
        agent=agent,
        messages=[ChatMessage(id=str(uuid.uuid4()), role="user", content=content)],
        thread_id=thread_id,
        user=ctx.principal,
    )
    if title:
        await ThreadService.update_thread(thread_id, title=title, user=ctx.principal)


def _as_text(payload: Any) -> str:
    """Render a step's output as message text, JSON rather than a Python repr."""
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, default=str, indent=2)
