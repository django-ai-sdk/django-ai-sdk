from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType, SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from django_ai_sdk.common import ChatMessage
from django_ai_sdk.errors import Blocked
from django_ai_sdk.workflows.registry import ADHOC

if TYPE_CHECKING:
    from collections.abc import Mapping
    from contextlib import AbstractAsyncContextManager

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.agent import Agent
    from django_ai_sdk.oversight.policy import Policy
    from django_ai_sdk.oversight.turns import OpenTurn
    from django_ai_sdk.workflows.listeners import RunListener


HISTORY_ROLES = frozenset({"user", "assistant"})


class WorkflowStopped(Exception):
    """A step failed and its `on_error` says the run stops there."""


class RunSuperseded(Exception):
    """This worker no longer holds the run; raised before writing over a newer attempt."""


class StepRecorded(Exception):
    def __init__(self, outcome: StepOutcome) -> None:
        super().__init__(outcome.detail)
        self.outcome = outcome


class OnError(StrEnum):
    STOP = "stop"
    CONTINUE = "continue"


@dataclass
class StepOutcome:
    output: Any = None
    status: Literal["completed", "failed", "skipped", "waiting"] = "completed"
    detail: str = ""
    turn_id: str = ""


@dataclass
class WorkflowContext:
    inputs: Mapping[str, Any] = field(default_factory=dict)
    steps: Mapping[str, StepOutcome] = field(default_factory=dict)
    user: AbstractBaseUser | AnonymousUser | None = None
    workflow: str = ""
    run_id: str = ""
    declared: Mapping[str, Step] = field(default_factory=dict, repr=False)

    def input(self, name: str, default: Any = None) -> Any:
        return self.inputs.get(name, default)

    def step(self, name: str, default: Any = None) -> Any:
        """What a step produced, or `default` unless it completed."""
        outcome = self.steps.get(name)
        if outcome is None or outcome.status != "completed":
            return default
        return outcome.output

    def approval(self, name: str) -> dict[str, Any] | None:
        """The approval `{approved, by, note}` step `name` completed with, or None."""
        outcome = self.steps.get(name)
        if not isinstance(self.declared.get(name), ApprovalStep) or outcome is None:
            return None
        decision = outcome.output
        if outcome.status != "completed" or not isinstance(decision, dict):
            return None
        return dict(decision) if decision.get("approved") is True else None


class Step:
    """At least once, not exactly once: external side effects must use `idempotency_key(ctx)`."""

    name: str = ""

    requires: tuple[str, ...] = ()

    on_error: OnError = OnError.STOP

    listeners: tuple[RunListener, ...] = ()

    config: Mapping[str, Any] = MappingProxyType({})

    waits: bool = False

    @classmethod
    def validate_config(cls, config: Mapping[str, Any]) -> None:
        """Raise ValueError unless this step type accepts `config` (default: none)."""
        if config:
            raise ValueError(f"{cls.__name__} takes no config, got keys {sorted(config)}.")

    async def skip_if(self, ctx: WorkflowContext) -> str:
        """A reason to skip, or "" to run."""
        return ""

    async def run(self, ctx: WorkflowContext) -> StepOutcome:
        raise NotImplementedError

    def idempotency_key(self, ctx: WorkflowContext) -> str:
        """ "<run id>:<step name>", stable across recovery and resume. Recorded runs only."""
        if not ctx.run_id or not self.name:
            raise RuntimeError("Only a step of a recorded run has an idempotency key.")
        return f"{ctx.run_id}:{self.name}"

    def waiting_for(self) -> str:
        return ""

    def wait_seconds(self) -> float:
        return 0.0

    def decide(self, decision: Any) -> StepOutcome:
        """The outcome for the decision that arrived, or for None when none came in time."""
        raise NotImplementedError

    def channel(self, ctx: WorkflowContext) -> str:
        """The channel for this step's turns, "workflow:<key>:<step>"."""
        return f"workflow:{ctx.workflow or ADHOC}:{self.name}"

    def turn(
        self,
        ctx: WorkflowContext,
        agent: Agent | type[Policy] | str,
        *,
        input: str | None = None,
        messages: list[ChatMessage] | None = None,
    ) -> AbstractAsyncContextManager[OpenTurn]:
        """Open a turn for a model this step calls directly, as the run's user."""
        from django_ai_sdk.oversight.policy import Policy
        from django_ai_sdk.oversight.turns import turn

        policy_only = (
            isinstance(agent, str) or isinstance(agent, type) and issubclass(agent, Policy)
        )
        step_as_agent = SimpleNamespace(name=self.name or type(self).__name__, agent_id="")
        return turn(
            step_as_agent if policy_only else agent,
            channel=self.channel(ctx),
            user=ctx.user,
            input=input,
            messages=messages,
            oversight=agent if policy_only else None,
        )


class AgentStep(Step):
    agent: type[Agent] | None = None
    agent_id: str = ""

    schema: type[BaseModel] | None = None

    instructions: str = ""

    history: tuple[str, ...] = ()

    async def get_agent(self) -> Agent:
        if self.agent is not None:
            return self.agent()
        from django_ai_sdk.agents.services import AgentService

        return await AgentService.get(self.agent_id)

    async def system_prompt(self, ctx: WorkflowContext) -> str | None:
        """The override this step declares, or None to keep the agent's own."""
        return self.instructions or None

    async def user_message(self, ctx: WorkflowContext) -> str:
        """By default, what the steps it requires produced."""
        parts = [
            f"[{name}]\n{_as_text(ctx.step(name))}"
            for name in self.requires
            if ctx.step(name) is not None
        ]
        return "\n\n".join(parts)

    def transcript(self, ctx: WorkflowContext) -> list[ChatMessage]:
        """The inputs named in `history`, as ChatMessages. User and assistant turns only."""
        transcript: list[ChatMessage] = []
        for name in self.history:
            value = ctx.input(name)
            if value is None:
                continue
            if isinstance(value, str | dict):
                value = [value]
            for item in value:
                message = _as_message(item)
                if message.role not in HISTORY_ROLES:
                    raise ValueError(
                        f"History input {name!r} holds a {message.role!r} message; history "
                        f"may only hold user and assistant turns."
                    )
                transcript.append(message)
        return transcript

    async def messages(self, ctx: WorkflowContext) -> list[ChatMessage]:
        turn = await self.user_message(ctx)
        history = self.transcript(ctx)
        return [*history, ChatMessage(role="user", content=turn)] if turn else history

    async def run(self, ctx: WorkflowContext) -> StepOutcome:
        from django_ai_sdk.permissions import Operation, check_permissions, get_agent_permissions

        agent = await self.get_agent()
        await check_permissions(ctx.user, Operation.CHAT, get_agent_permissions(agent), agent=agent)
        response_format = {"response_format": self.schema} if self.schema else {}
        messages = await self.messages(ctx)

        try:
            result = await agent.run(
                messages,
                system_prompt=await self.system_prompt(ctx),
                user=ctx.user,
                channel=self.channel(ctx),
                **response_format,
            )
        except Blocked as blocked:
            done = "blocked" if blocked.stage == "gate" else "withheld"
            return StepOutcome(status="failed", detail=f"{done} by oversight: {blocked.message}")
        if result is None:
            return StepOutcome(status="failed", detail="the agent returned nothing")
        return self.outcome_for(result)

    def outcome_for(self, result: Any) -> StepOutcome:
        """The agent's result as this step's outcome; a missing declared schema is a failure."""
        if isinstance(result, BaseModel):
            return StepOutcome(output=result.model_dump(mode="json"))
        if self.schema is not None:
            return StepOutcome(status="failed", detail="the agent returned no structured output")
        return StepOutcome(output=result)


class ApprovalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(default="", max_length=255)
    timeout_seconds: int = Field(default=7 * 24 * 3600, ge=60, le=90 * 24 * 3600)


class ApprovalStep(Step):
    """Wait (durably, on DBOS) for a person to approve the run before later steps.

    Rejection or timeout fails the step; nobody decides on a run they started.
    """

    waits = True

    @classmethod
    def validate_config(cls, config: Mapping[str, Any]) -> None:
        try:
            ApprovalConfig.model_validate(dict(config))
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc

    @property
    def settings(self) -> ApprovalConfig:
        return ApprovalConfig.model_validate(dict(self.config))

    def waiting_for(self) -> str:
        return self.settings.prompt or "Waiting for approval"

    def wait_seconds(self) -> float:
        return float(self.settings.timeout_seconds)

    async def run(self, ctx: WorkflowContext) -> StepOutcome:
        return StepOutcome(
            status="failed", detail="an approval step waits only in a durable run (DBOS)"
        )

    def decide(self, decision: Any) -> StepOutcome:
        if not isinstance(decision, dict):
            return StepOutcome(status="failed", detail="no decision came in time")
        by = str(decision.get("by") or "")
        output = {
            "approved": bool(decision.get("approved")),
            "by": by,
            "note": str(decision.get("note") or ""),
        }
        if output["approved"]:
            return StepOutcome(output=output, detail=f"approved by {by}"[:255])
        note = f": {output['note']}" if output["note"] else ""
        return StepOutcome(status="failed", output=output, detail=f"rejected by {by}{note}"[:255])


def _as_message(item: Any) -> ChatMessage:
    if isinstance(item, ChatMessage):
        return item
    if isinstance(item, str):
        return ChatMessage(role="user", content=item)
    return ChatMessage.model_validate(item)


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, indent=2)


__all__ = [
    "AgentStep",
    "ApprovalConfig",
    "ApprovalStep",
    "OnError",
    "RunSuperseded",
    "Step",
    "StepOutcome",
    "StepRecorded",
    "WorkflowContext",
    "WorkflowStopped",
]
