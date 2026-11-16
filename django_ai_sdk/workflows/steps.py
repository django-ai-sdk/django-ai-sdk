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


# Roles allowed in a step's history. Only the agent writes system turns.
HISTORY_ROLES = frozenset({"user", "assistant"})


class WorkflowStopped(Exception):
    """A step failed and its `on_error` says the run stops there."""


class RunSuperseded(Exception):
    """This worker no longer holds the run. It was swept, resumed or claimed again.

    `RunRecorder` raises it before writing, so a stale worker stops instead of
    overwriting the newer attempt.
    """


class StepRecorded(Exception):
    """The step's row already holds this attempt's outcome: the step does not run again.

    `RunRecorder` raises it when a step finished and was recorded, but its worker died
    before DBOS checkpointed it. The runner takes `outcome` as the step's.
    """

    def __init__(self, outcome: StepOutcome) -> None:
        super().__init__(outcome.detail)
        self.outcome = outcome


class OnError(StrEnum):
    """What a step's failure does to the rest of the run."""

    # Abort the run.
    STOP = "stop"
    # Record it, skip this step's dependents, run the rest.
    CONTINUE = "continue"


@dataclass
class StepOutcome:
    """What a step reports back, filed under the step's own name."""

    output: Any = None
    # "waiting" only while an approval step waits for a decision.
    status: Literal["completed", "failed", "skipped", "waiting"] = "completed"
    # One line on the outcome, for whoever reads the record.
    detail: str = ""
    # The oversight turn the step opened, if any. The runner fills it in.
    turn_id: str = ""


@dataclass
class WorkflowContext:
    """The run's state, as a step or a listener sees it.

    Two namespaces, so nothing collides: `inputs` is what the caller supplied,
    `steps` is what has run so far, keyed by step name.
    """

    inputs: Mapping[str, Any] = field(default_factory=dict)
    steps: Mapping[str, StepOutcome] = field(default_factory=dict)
    user: AbstractBaseUser | AnonymousUser | None = None
    # The run's workflow key (the stored workflow's slug, the registered workflow's
    # name, or "adhoc") and its id, for listeners that report on the run.
    workflow: str = ""
    run_id: str = ""
    # The run's steps by name, so `approval` knows which one is an approval step.
    declared: Mapping[str, Step] = field(default_factory=dict, repr=False)

    def input(self, name: str, default: Any = None) -> Any:
        """One of the run's declared inputs."""
        return self.inputs.get(name, default)

    def step(self, name: str, default: Any = None) -> Any:
        """What a step produced, or `default` unless it completed."""
        outcome = self.steps.get(name)
        if outcome is None or outcome.status != "completed":
            return default
        return outcome.output

    def approval(self, name: str) -> dict[str, Any] | None:
        """The approval `{approved, by, note}` step `name` completed with, or None.

        Check an approval with this, not with `step(name)`: any step can be named
        "approve" and produce `{"approved": true}`, but only an `ApprovalStep`
        completes by a person's decision through `WorkflowService.decide`, and only
        when they approved. None for any other step, a rejection, a timeout, or a
        step that did not run.
        """
        outcome = self.steps.get(name)
        if not isinstance(self.declared.get(name), ApprovalStep) or outcome is None:
            return None
        decision = outcome.output
        if outcome.status != "completed" or not isinstance(decision, dict):
            return None
        return dict(decision) if decision.get("approved") is True else None


class Step:
    """One unit of work in a workflow.

    A step runs once every step it `requires` has completed, and its result is
    filed under `name`.

    A step runs at least once, not exactly once. A worker that dies mid-step, or
    before DBOS records the step as done, runs it again when the run is recovered,
    and a resumed run runs again every step that did not complete. A step with a
    side effect outside the run's own rows (a payment, a booking, a sent message)
    must make it idempotent by `idempotency_key(ctx)`: pass it to the API as its
    idempotency key, or store it with the effect and do nothing when it is there.
    """

    # The step's key in the run's record, and the name later steps require.
    name: str = ""

    requires: tuple[str, ...] = ()

    on_error: OnError = OnError.STOP

    listeners: tuple[RunListener, ...] = ()

    # The StepDefinition's `config`, checked by `validate_config` before it is
    # stored. Empty for a step written in code.
    config: Mapping[str, Any] = MappingProxyType({})

    # True for a step that waits for a person (ApprovalStep). Only a durable run waits.
    waits: bool = False

    @classmethod
    def validate_config(cls, config: Mapping[str, Any]) -> None:
        """Raise ValueError unless this step type accepts `config`.

        Called when a definition is stored or registered, so a bad config fails
        early instead of in the worker. By default a step takes no config. Override
        this to accept settings, for example with a pydantic model's `model_validate`.
        """
        if config:
            raise ValueError(f"{cls.__name__} takes no config, got keys {sorted(config)}.")

    async def skip_if(self, ctx: WorkflowContext) -> str:
        """A reason to skip, or "" to run.

        A step having completed proves it produced something, never that the value
        is usable, so a precondition that needs looking at goes here.
        """
        return ""

    async def run(self, ctx: WorkflowContext) -> StepOutcome:
        """Do the work and report the outcome. It may run more than once (see the class)."""
        raise NotImplementedError

    def idempotency_key(self, ctx: WorkflowContext) -> str:
        """ "<run id>:<step name>": the same every time this step of this run runs.

        It stays the same across recovery and resume, so an effect done under it is
        done once however often the step runs. Only a recorded run has one.
        """
        if not ctx.run_id or not self.name:
            raise RuntimeError("Only a step of a recorded run has an idempotency key.")
        return f"{ctx.run_id}:{self.name}"

    # A waiting step (`waits = True`) implements these three instead of doing work in run().

    def waiting_for(self) -> str:
        """One line on what the step waits for, shown while it waits."""
        return ""

    def wait_seconds(self) -> float:
        """How long the step waits for its decision before it fails."""
        return 0.0

    def decide(self, decision: Any) -> StepOutcome:
        """The outcome for the decision that arrived, or for None when none came in time."""
        raise NotImplementedError

    def channel(self, ctx: WorkflowContext) -> str:
        """The channel for this step's turns, "workflow:<key>:<step>".

        The key is the stored workflow's slug or the registered workflow's name. It is
        "adhoc" for any other definition, or a pipeline run with no workflow key.
        """
        return f"workflow:{ctx.workflow or ADHOC}:{self.name}"

    def turn(
        self,
        ctx: WorkflowContext,
        agent: Agent | type[Policy] | str,
        *,
        input: str | None = None,
        messages: list[ChatMessage] | None = None,
    ) -> AbstractAsyncContextManager[OpenTurn]:
        """Open a turn for a model this step calls directly, as the run's user.

        Every model call in a step must happen inside a turn (`oversight.calls`). An
        AgentStep's `agent.run()` opens one itself. Use this for any other model
        call, under `agent`'s policy or a named policy::

            async with self.turn(ctx, "translation", input=chunk) as t:
                text = await t.finish(await translate(await t.gate()))
        """
        from django_ai_sdk.oversight.turns import turn

        # Oversight: a policy given without an agent judges the turn as this step.
        policy_only = isinstance(agent, str) or isinstance(agent, type) and _is_policy(agent)
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

        Only user and assistant turns are allowed. Inputs come from whoever started the
        run, and a system turn would speak with the agent's authority. A step that
        needs one writes it in `system_prompt()`.
        """
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
        await check_permissions(ctx.user, Operation.CHAT, get_agent_permissions(agent), agent=agent)
        # Omitted rather than passed as None: Agent.run reads an explicit None as
        # "no structured output", which would strip the agent's own.
        response_format = {"response_format": self.schema} if self.schema else {}
        messages = await self.messages(ctx)

        try:
            # Oversight: a workflow step is a turn like any other, under the agent's policy.
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
        """What the agent's result means as this step's outcome.

        A declared schema that did not come back is a failure, so `on_error` decides
        what it costs. A model that did is dumped to JSON, since it goes on the record.
        """
        if isinstance(result, BaseModel):
            return StepOutcome(output=result.model_dump(mode="json"))
        if self.schema is not None:
            return StepOutcome(status="failed", detail="the agent returned no structured output")
        return StepOutcome(output=result)


class ApprovalConfig(BaseModel):
    """An approval step's `config`."""

    model_config = ConfigDict(extra="forbid")

    # What the approver is asked, shown while the step waits.
    prompt: str = Field(default="", max_length=255)
    # How long the run waits for a decision before the step fails. A week by default.
    timeout_seconds: int = Field(default=7 * 24 * 3600, ge=60, le=90 * 24 * 3600)


class ApprovalStep(Step):
    """Wait for a person to approve the run before the steps after it.

    The run waits durably, across restarts, so it needs DBOS (`django_ai_sdk.durable`).
    The decision comes from `WorkflowService.decide`, and nobody decides on a run they
    started. Approved, the step completes with `{approved, by, note}`. Rejected, or
    without a decision in time, it fails, and its `on_error` decides what follows.
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
        # Reached only by a run that cannot wait: an in-process one.
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
    """One history item as a ChatMessage, whatever it crossed the queue as."""
    if isinstance(item, ChatMessage):
        return item
    if isinstance(item, str):
        return ChatMessage(role="user", content=item)
    return ChatMessage.model_validate(item)


def _is_policy(cls: type) -> bool:
    from django_ai_sdk.oversight.policy import Policy

    return issubclass(cls, Policy)


def _as_text(value: Any) -> str:
    """A step's output as prompt text, JSON rather than a Python repr."""
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
