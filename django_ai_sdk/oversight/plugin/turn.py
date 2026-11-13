"""The run's turn: what the caller asks of it, and what oversight decided.

`run_agent` opens the turn before the run, or joins the one it runs inside, and
closes it after, also when the run raises. The run's `hook_context` carries the
`TurnRequest`, so the hooks read and fill in the same object the caller holds.

A turn inside another one joins it when its policy is the same (or there is
none). Another policy opens a child turn, linked to the outer one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from haystack.core.serialization import default_from_dict, default_to_dict

from django_ai_sdk.errors import Blocked
from django_ai_sdk.oversight.guard import TurnGuard, current_guard, outlived_turn
from django_ai_sdk.oversight.guard.state import merge_restrict
from django_ai_sdk.oversight.types import Stage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from haystack.components.agents.state.state import State
    from pydantic import BaseModel

    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.oversight.guard import FinishResult, GateResult
    from django_ai_sdk.oversight.plugin.finish import WholeRun
    from django_ai_sdk.oversight.policy import Policy

# Where the run's hook_context holds its TurnRequest.
REQUEST_KEY = "django_ai_sdk.oversight"


class _UnnamedAgent:
    """The agent of a Haystack run that names none: it has no policy of its own.

    So it runs under DEFAULT_POLICY, like any agent without one, and is recorded
    under this name. Without DEFAULT_POLICY it opens no turn.
    """

    name = "Haystack agent"
    oversight = None


UNNAMED_AGENT = _UnnamedAgent()


@dataclass
class TurnRequest:
    """A run's turn: what the caller wants of it, then what oversight decided.

    Pass it to `run_agent(request=...)`. Once the run started, `guard` is the
    turn it opened, or None when it joined an outer turn or has no policy.
    """

    # The SDK agent whose policy the run is under.
    agent: Any = None
    channel: str = "run"
    user: Any = None
    thread_id: str | None = None
    # A policy for this run only, as in `Agent.run(oversight=...)`.
    oversight: type[Policy] | str | Literal[False] | None = None
    # The stored thread vouches for the history, so the gate does not judge it again.
    history_vouched: bool = False
    # Called once the gate passed the question, such as to store it.
    on_gated: Callable[[TurnRequest], Awaitable[None]] | None = None
    # What finish judges and replaces. All the text the model wrote by default.
    answer: WholeRun | None = None
    # A Pydantic model the answer must still fit once masked.
    schema: type[BaseModel] | None = None
    # The id of the message that will store the answer.
    message_id: str = ""
    # The caller closes the turn itself, such as a stream that still stores the answer.
    keep_open: bool = False
    # Called with the stubs of refused tool calls, which never reach the ToolInvoker.
    on_tool_results: Callable[[list[Any]], Awaitable[None]] | None = None
    guard: TurnGuard | None = None
    gated: GateResult | None = None
    # Index of the first message the run produced, after the gate.
    first: int = 0

    @property
    def refused(self) -> bool:
        return self.gated is not None and self.gated.blocked

    @property
    def finished(self) -> FinishResult | None:
        return self.guard.finished if self.guard is not None else None

    def raise_if_blocked(self) -> None:
        """Raise `Blocked` when the gate refused the question or finish withheld the answer."""
        if self.guard is None:
            return
        if self.refused:
            assert self.gated is not None
            raise Blocked(self.gated.message, stage=str(Stage.GATE), turn_id=self.guard.id)
        if (finished := self.finished) is not None and finished.blocked:
            raise Blocked(finished.message, stage=str(Stage.FINISH), turn_id=self.guard.id)

    def context(self, **extra: Any) -> dict[str, Any]:
        """A hook_context holding this request, beside `extra` for other hooks."""
        return {**extra, REQUEST_KEY: self}


class OversightHook:
    """Base for the oversight hooks: serializable, so an Agent with them still saves to YAML.

    Haystack calls `run_async` in an async run, and `run` in a sync one. A sync
    run never opens a turn of its own (it cannot await the guard), so `run`
    does nothing unless a hook overrides it.
    """

    allowed_hook_points: list[str] = []  # noqa: RUF012

    def run(self, state: State) -> None:
        """A sync run: nothing to do."""

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OversightHook:
        return default_from_dict(cls, data)


class OversightStopHook(OversightHook):
    """Before each model call: stop a run whose turn has closed, and offer only the tools it allows."""

    allowed_hook_points = ["before_llm"]  # noqa: RUF012

    def run(self, state: State) -> None:
        if outlived_turn():
            state.set("stop_run", "turn closed")
        elif (guard := current_guard()) is not None:
            # Haystack hands this same list to the model and the ToolInvoker for
            # the step, so filtering it in place takes the tools away. The tool
            # hook refuses a call to a removed tool anyway.
            tools = state.data["tools"]
            tools[:] = [tool for tool in tools if guard.allows_tool(tool.name)]

    async def run_async(self, state: State) -> None:
        self.run(state)


def request_in(state: State) -> TurnRequest:
    """The run's TurnRequest. A run started without one gets the defaults."""
    return (state.data.get("hook_context") or {}).get(REQUEST_KEY) or TurnRequest()


def agent_of(request: TurnRequest) -> Any:
    """The agent whose policy the run is under. Named by nothing, outside a turn:
    the unnamed agent, under DEFAULT_POLICY. Inside one, the run joins it."""
    if request.agent is None and current_guard() is None:
        return UNNAMED_AGENT
    return request.agent


async def open_turn(
    agent: Any,
    *,
    channel: str,
    user: Any = None,
    thread_id: str | None = None,
    messages: list[ChatMessage] | None = None,
    input: str | None = None,
    oversight: type[Policy] | str | Literal[False] | None = None,
) -> tuple[TurnGuard, bool]:
    """Start a turn of `agent`. Returns its guard, and whether it opened a turn of its own.

    It opens none without a policy, or inside a turn of the same policy, which
    it joins. Inside a turn of another policy it opens a child turn, which is
    still the outer turn's person and keeps the tools the outer turn took away.
    """
    outer = current_guard()
    guard = await TurnGuard.start(
        agent,
        channel=channel,
        user=user,
        thread_id=thread_id,
        messages=messages,
        input=input,
        oversight=oversight,
    )
    if not guard.active or joins_turn(outer, guard.resolved.name):
        return guard, False
    if outer is not None:
        guard.parent_turn_id = outer.id
        guard.actor_id = guard.actor_id or outer.actor_id
        if outer.restrict is not None:
            guard.restrict = merge_restrict(guard.restrict, outer.restrict)
    return guard, True


def joins_turn(outer: TurnGuard | None, policy_name: str) -> bool:
    """Whether a run under `policy_name` joins the open turn `outer` instead of its own."""
    return outer is not None and outer.policy is not None and outer.policy.name == policy_name


def needs_own_turn(request: TurnRequest) -> bool:
    """Whether the run needs a turn of its own here, which only an async run opens.

    Sync code cannot await `get_oversight()`, so an agent that overrides it is
    assumed to need one, and so is a policy name that does not resolve.
    """
    from django_ai_sdk.oversight.agents import code_policy, policy_needs_await
    from django_ai_sdk.oversight.conf import is_enabled
    from django_ai_sdk.oversight.registry import policy_name

    agent, oversight = agent_of(request), request.oversight
    if agent is None or oversight in (False, "") or not is_enabled():
        return False
    try:
        if oversight:
            name = policy_name(oversight)
        elif policy_needs_await(agent):
            return True
        else:
            name = code_policy(agent)[0]
    except Exception:
        return True
    return bool(name) and not joins_turn(current_guard(), name)


async def opens_own_turn(request: TurnRequest) -> bool:
    """Whether the run of `request.agent` opens a turn of its own here. True when its policy cannot be told."""
    from django_ai_sdk.oversight.agents import acode_policy
    from django_ai_sdk.oversight.conf import is_enabled
    from django_ai_sdk.oversight.registry import policy_name

    agent, oversight = request.agent, request.oversight
    if agent is None or oversight in (False, "") or not is_enabled():
        return False
    try:
        name = policy_name(oversight) if oversight else ""
        name = name or (await acode_policy(agent, channel=request.channel, user=request.user))[0]
    except Exception:
        return True
    return bool(name) and not joins_turn(current_guard(), name)


__all__ = [
    "REQUEST_KEY",
    "UNNAMED_AGENT",
    "OversightHook",
    "OversightStopHook",
    "TurnRequest",
    "agent_of",
    "joins_turn",
    "needs_own_turn",
    "open_turn",
    "opens_own_turn",
    "request_in",
]
