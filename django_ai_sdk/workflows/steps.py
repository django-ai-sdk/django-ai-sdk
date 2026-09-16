"""What a workflow step is, and the context every step and hook reads.

A step is a named unit of work that reads the run's state and returns a value:
`AgentStep` calls a model, a plain `Step` is arbitrary Python. A step's result is
filed under its own `name`, so two steps never contend for one key.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from django_ai_sdk.common import ChatMessage

if TYPE_CHECKING:
    from collections.abc import Mapping

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.agent import Agent
    from django_ai_sdk.workflows.hooks import WorkflowHook


class StepFailed(Exception):
    """A step failed and its `on_error` says the run stops there."""


class StepAlreadyRunning(Exception):
    """Another run holds this step. Raised by a hook whose rows are also a claim."""


class OnError(StrEnum):
    """What a step's failure does to the rest of the run."""

    # Abort the run.
    FAIL = "fail"
    # Record it, skip this step's dependents, run the rest.
    CONTINUE = "continue"


@dataclass
class StepOutcome:
    """What a step reports back, filed under the step's own name."""

    output: Any = None
    status: Literal["completed", "failed", "skipped"] = "completed"
    # One line on the outcome, for whoever reads the record.
    detail: str = ""


@dataclass
class WorkflowContext:
    """The run's state, as a step or a hook sees it.

    Two namespaces, so nothing collides: `inputs` is what the caller supplied,
    `steps` is what has run so far, keyed by step name.
    """

    inputs: Mapping[str, Any] = field(default_factory=dict)
    steps: Mapping[str, StepOutcome] = field(default_factory=dict)
    principal: AbstractBaseUser | AnonymousUser | None = None
    # The definition's name and the run's id, for hooks that report on the run.
    workflow: str = ""
    run_id: str = ""

    def input(self, name: str, default: Any = None) -> Any:
        """One of the run's declared inputs."""
        return self.inputs.get(name, default)

    def step(self, name: str, default: Any = None) -> Any:
        """What a step produced, or `default` unless it completed."""
        outcome = self.steps.get(name)
        if outcome is None or outcome.status != "completed":
            return default
        return outcome.output


class Step:
    """One unit of work in a workflow.

    A step runs once every step it `requires` has completed, and its result is
    filed under `name`.
    """

    # The step's key in the run's record, and the name later steps require.
    name: str = ""

    requires: tuple[str, ...] = ()

    on_error: OnError = OnError.FAIL

    hooks: tuple[WorkflowHook, ...] = ()

    async def skip_when(self, ctx: WorkflowContext) -> str:
        """A reason to skip, or "" to run.

        A step having completed proves it produced something, never that the value
        is usable, so a precondition that needs looking at goes here.
        """
        return ""

    async def run(self, ctx: WorkflowContext) -> StepOutcome:
        """Do the work and report the outcome."""
        raise NotImplementedError


class AgentStep(Step):
    """One structured agent call, as one step among others.

    `agent` is a class a code-declared step imports; `agent_id` is the id a JSON
    definition carries.
    """

    # Exactly one of these identifies the agent.
    agent: type[Agent] | None = None
    agent_id: str = ""

    # Declared output_fields, compiled. None means the agent's own response format.
    schema: type[BaseModel] | None = None

    # A system prompt replacing the agent's own; "" keeps the agent's.
    instructions: str = ""

    # The inputs to send as the conversation before this step's own turn. A
    # definition names its input fields per step; a code step names them itself.
    history: tuple[str, ...] = ()

    async def get_agent(self) -> Agent:
        """The agent this step calls, from the class or the id."""
        if self.agent is not None:
            return self.agent()
        from django_ai_sdk.agents.services import AgentService

        return await AgentService.get(self.agent_id)

    async def system_prompt(self, ctx: WorkflowContext) -> str | None:
        """The override this step declares, or None to keep the agent's own."""
        return self.instructions or None

    async def user_message(self, ctx: WorkflowContext) -> str:
        """The turn this step asks. By default, what the steps it requires produced."""
        parts = [
            f"[{name}]\n{_as_text(ctx.step(name))}"
            for name in self.requires
            if ctx.step(name) is not None
        ]
        return "\n\n".join(parts)

    def transcript(self, ctx: WorkflowContext) -> list[ChatMessage]:
        """The inputs named in `history`, as ChatMessages.

        What a history field holds is the author's choice: a bare string is one
        user turn, a message dict is one message, and a list is a conversation
        whose items are messages, message dicts, or strings.
        """
        transcript: list[ChatMessage] = []
        for name in self.history:
            value = ctx.input(name)
            if value is None:
                continue
            if isinstance(value, str):
                transcript.append(ChatMessage(role="user", content=value))
            elif isinstance(value, dict):
                transcript.append(ChatMessage.model_validate(value))
            else:
                transcript.extend(_as_message(item) for item in value)
        return transcript

    async def messages(self, ctx: WorkflowContext) -> list[ChatMessage]:
        """The run's transcript, then this step's own turn."""
        turn = await self.user_message(ctx)
        history = self.transcript(ctx)
        if not turn:
            return history
        return [*history, ChatMessage(role="user", content=turn)]

    async def run(self, ctx: WorkflowContext) -> StepOutcome:
        """Call the agent with this step's prompt and messages."""
        from django_ai_sdk.permissions import Operation, check_permissions, get_agent_permissions

        agent = await self.get_agent()
        # SECURITY: composing an agent is not permission to run it, and a workflow
        # is not a bypass.
        await check_permissions(
            ctx.principal, Operation.CHAT, get_agent_permissions(agent), agent=agent
        )
        # Omitted rather than passed as None: Agent.run reads an explicit None as
        # "no structured output", which would strip the agent's own.
        response_format = {"response_format": self.schema} if self.schema else {}
        result = await agent.run(
            await self.messages(ctx),
            system_prompt=await self.system_prompt(ctx),
            user=ctx.principal,
            **response_format,
        )
        if result is None:
            return StepOutcome(status="failed", detail="the agent returned nothing")
        return self.outcome_for(result)

    def outcome_for(self, result: Any) -> StepOutcome:
        """What the agent's result means as this step's outcome.

        A declared schema that did not come back is a failure, so `on_error` decides
        what it costs. A model that did is dumped to JSON, since it goes on the record.
        """
        if isinstance(result, BaseModel):
            return StepOutcome(output=result.model_dump(mode="json"))
        if self.schema is not None:
            return StepOutcome(status="failed", detail="the agent returned no structured output")
        return StepOutcome(output=result)


def _as_message(item: Any) -> ChatMessage:
    """One history item as a ChatMessage, whatever it crossed the queue as."""
    if isinstance(item, ChatMessage):
        return item
    if isinstance(item, str):
        return ChatMessage(role="user", content=item)
    return ChatMessage.model_validate(item)


def _as_text(value: Any) -> str:
    """A step's output as prompt text, JSON rather than a Python repr."""
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, indent=2)


__all__ = [
    "AgentStep",
    "OnError",
    "Step",
    "StepAlreadyRunning",
    "StepFailed",
    "StepOutcome",
    "WorkflowContext",
]
