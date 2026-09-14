"""What a workflow step is.

A step is a named unit of work that reads the run's state and returns a value:
`AgentStep` calls a model, a plain `Step` is arbitrary Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser
    from pydantic import BaseModel

    from django_ai_sdk.agent import Agent


class StepFailed(Exception):
    """A step failed and its `on_error` says the run stops there."""


class StepAlreadyRunning(Exception):
    """Another run holds this step. Raised from a sink whose rows are also a claim."""


class OnError(StrEnum):
    """What a step's failure does to the rest of the run."""

    # Abort the run.
    FAIL = "fail"
    # Record it, skip this step's dependents, run the rest.
    CONTINUE = "continue"


@dataclass
class StepContext:
    """The run's state, as one step sees it.

    `inputs` is what the caller passed in; `outputs` is what earlier steps
    produced, keyed by the names they declared in `provides`.
    """

    inputs: Mapping[str, Any] = field(default_factory=dict)
    outputs: Mapping[str, Any] = field(default_factory=dict)
    principal: AbstractBaseUser | AnonymousUser | None = None

    def get(self, name: str, default: Any = None) -> Any:
        """A value from the run's state, outputs before inputs."""
        if name in self.outputs:
            return self.outputs[name]
        return self.inputs.get(name, default)


@dataclass
class StepOutcome:
    """What a step reports back.

    `output` is filed under the name the step declares in `provides`, and
    ignored when it declares none.
    """

    output: Any = None
    status: Literal["completed", "failed", "skipped"] = "completed"
    # One line on the outcome, for whoever reads the record.
    detail: str = ""


class Step:
    """One unit of work in a workflow.

    A step runs once every name it `requires` is on the run's state, and files
    what it returns under the name it `provides`.
    """

    # The step's key in the run's record. Opaque to the runner.
    name: str = ""

    requires: tuple[str, ...] = ()
    # The name this step's output is filed under, or "" to publish nothing.
    provides: str = ""

    # When set and on_error is CONTINUE, a failure payload is filed under this name.
    error_key: str = ""

    on_error: OnError = OnError.FAIL

    async def skip_when(self, ctx: StepContext) -> str:
        """A reason to skip, or "" to run.

        A name being on the table proves it was produced, never that its value is
        usable, so a precondition that needs looking at goes here.
        """
        return ""

    async def run(self, ctx: StepContext) -> StepOutcome:
        """Do the work and report the outcome."""
        raise NotImplementedError


class AgentStep(Step):
    """One structured agent call, as one step among others."""

    agent: type[Agent]
    schema: type[BaseModel] | None = None

    async def system_prompt(self, ctx: StepContext) -> str:
        """The step's instructions."""
        return self.agent().get_system_prompt()

    async def user_message(self, ctx: StepContext) -> str:
        """The turn this step asks the agent."""
        raise NotImplementedError

    async def messages(self, ctx: StepContext) -> list[Any]:
        """The conversation this step sends."""
        from django_ai_sdk.common import ChatMessage

        return [ChatMessage(role="user", content=await self.user_message(ctx))]

    async def run_agent(
        self,
        ctx: StepContext,
        agent: Agent,
        messages: list[Any],
        system_prompt: str | None,
    ) -> StepOutcome:
        """Check the CHAT gate, call the agent, and read what came back."""
        from django_ai_sdk.permissions import Operation, check_permissions, get_agent_permissions

        # SECURITY: composing an agent into a step is not permission to run it.
        await check_permissions(
            ctx.principal, Operation.CHAT, get_agent_permissions(agent), agent=agent
        )
        # Omitted rather than passed as None: Agent.run reads an explicit None as
        # "no structured output", which would strip the agent's own.
        schema = {"response_format": self.schema} if self.schema else {}
        result = await agent.run(
            messages,
            system_prompt=system_prompt,
            user=ctx.principal,
            **schema,
        )
        if result is None:
            return StepOutcome(status="failed", detail="the agent returned nothing")
        return self.outcome_for(result)

    def outcome_for(self, result: Any) -> StepOutcome:
        """What the agent's result means as this step's outcome."""
        return StepOutcome(output=result)

    async def run(self, ctx: StepContext) -> StepOutcome:
        """Call the agent with this step's prompt and messages."""
        return await self.run_agent(
            ctx, self.agent(), await self.messages(ctx), await self.system_prompt(ctx)
        )


__all__ = [
    "AgentStep",
    "OnError",
    "Step",
    "StepAlreadyRunning",
    "StepContext",
    "StepFailed",
    "StepOutcome",
]
