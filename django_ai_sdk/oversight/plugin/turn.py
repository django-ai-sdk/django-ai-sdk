"""The turn, carried by the run: opened in before_run, closed in after_run.

The run's `hook_context` carries a `TurnRequest`: which agent, channel, person
and thread the turn is for, and what the caller wants done along the way. The
hooks put the `Turn` on it, so the caller reads what oversight decided once the
run returns (see `turn_of`).

Haystack skips after_run when a run raises or is cancelled. `run_agent` closes
the turn then, with the error. A run started without it leaves its turn open,
written ahead after the gate, until the sweep marks it interrupted.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from haystack.core.serialization import default_from_dict, default_to_dict

from django_ai_sdk.errors import Blocked
from django_ai_sdk.oversight.agent_run import open_turn
from django_ai_sdk.oversight.guard import (
    TurnGuard,
    current_guard,
    enter,
    leave,
    outlived_turn,
)
from django_ai_sdk.oversight.types import Stage

if TYPE_CHECKING:
    import contextvars
    from collections.abc import Awaitable, Callable

    from haystack.components.agents.state.state import State
    from pydantic import BaseModel

    from django_ai_sdk.oversight.guard import FinishResult, GateResult
    from django_ai_sdk.oversight.plugin.finish import Answer
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
class Turn:
    """The turn a run belongs to. `owns` is False when it joined an outer turn or has no policy."""

    guard: TurnGuard
    owns: bool
    gated: GateResult | None = None
    # Index of the first message the run produced, after the gate.
    first: int = 0
    token: contextvars.Token[TurnGuard | None] | None = None
    # The agent of a run that joined an outer turn, named on each tool call it records.
    agent_name: str = ""

    @property
    def refused(self) -> bool:
        return self.gated is not None and self.gated.blocked

    @property
    def finished(self) -> FinishResult | None:
        return self.guard.finished if self.owns else None

    def raise_if_blocked(self) -> None:
        """Raise `Blocked` when the gate refused the question or finish withheld the answer."""
        if self.refused:
            assert self.gated is not None
            raise Blocked(self.gated.message, stage=str(Stage.GATE), turn_id=self.guard.id)
        if (finished := self.finished) is not None and finished.blocked:
            raise Blocked(finished.message, stage=str(Stage.FINISH), turn_id=self.guard.id)


@dataclass
class TurnRequest:
    """What the oversight hooks need to know about a run. Pass it as `hook_context=request.context()`.

    The hooks set `turn` once the run started.
    """

    # The SDK agent whose policy the run is under, unless TurnOpenHook names one.
    agent: Any = None
    channel: str = "run"
    user: Any = None
    thread_id: str | None = None
    # A policy for this run only, as in `Agent.run(oversight=...)`.
    oversight: type[Policy] | str | Literal[False] | None = None
    # The stored thread vouches for the history, so the gate does not judge it again.
    history_vouched: bool = False
    # Called once the gate passed the question, such as to store it.
    on_gated: Callable[[Turn | None], Awaitable[None]] | None = None
    # What finish judges and replaces. All the text the model wrote by default.
    answer: Answer | None = None
    # A Pydantic model the answer must still fit once masked.
    schema: type[BaseModel] | None = None
    # The id of the message that will store the answer.
    message_id: str = ""
    # The caller closes the turn itself, such as a stream that still stores the answer.
    keep_open: bool = False
    # Called with the stubs of refused tool calls, which never reach the ToolInvoker.
    on_tool_results: Callable[[list[Any]], Awaitable[None]] | None = None
    turn: Turn | None = None

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


class TurnOpenHook(OversightHook):
    """Open the run's turn under `agent`'s policy, or join the turn it runs inside (see `open_turn`).

    Without `agent`, the run's TurnRequest names it.
    """

    allowed_hook_points = ["before_run"]  # noqa: RUF012

    def __init__(self, agent: Any = None) -> None:
        self.agent = agent

    def to_dict(self) -> dict[str, Any]:
        cls = type(self.agent) if self.agent is not None else None
        return default_to_dict(self, agent=f"{cls.__module__}.{cls.__qualname__}" if cls else None)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TurnOpenHook:
        from haystack.core.serialization import import_class_by_name

        path = data.get("init_parameters", {}).get("agent")
        return cls(import_class_by_name(path)() if path else None)

    def run(self, state: State) -> None:
        """A sync run cannot open a turn: it joins the current one, if any."""
        request = request_in(state)
        outer = current_guard()
        request.turn = Turn(
            outer or TurnGuard(), owns=False, agent_name=self._joining(request, outer)
        )

    async def run_async(self, state: State) -> None:
        # A run given no TurnRequest was started outside the SDK, which always passes one.
        unnamed = request_of(state.data.get("hook_context")) is None
        request = request_in(state)
        agent = self.agent or request.agent
        outer = current_guard()
        if agent is None and unnamed and outer is None:
            # Nothing names the agent: the policy for agents without one applies.
            agent = UNNAMED_AGENT
        if agent is None:
            request.turn = Turn(outer or TurnGuard(), owns=False)
            return
        guard, opened = await open_turn(
            agent,
            channel=request.channel,
            user=request.user,
            thread_id=request.thread_id,
            input=_question(state),
            oversight=request.oversight,
        )
        if not opened:
            name = self._joining(request, outer)
            request.turn = Turn(outer or guard, owns=False, agent_name=name)
            return
        guard.message_id = request.message_id or guard.message_id
        guard.on_tool_results = request.on_tool_results
        request.turn = Turn(guard, owns=True, token=enter(guard))

    def _joining(self, request: TurnRequest, outer: TurnGuard | None) -> str:
        """The name of the agent joining `outer`, or "" when there is no outer turn."""
        agent = self.agent or request.agent
        if outer is None or agent is None:
            return ""
        return str(getattr(agent, "name", "") or type(agent).__name__)


class TurnCloseHook(OversightHook):
    """Close and record the run's turn, last thing in the run, unless the request keeps it open."""

    allowed_hook_points = ["after_run"]  # noqa: RUF012

    async def run_async(self, state: State) -> None:
        request = request_in(state)
        turn = request.turn
        _leave(turn)
        if turn is not None and turn.owns and not request.keep_open:
            await turn.guard.close()


class OversightStopHook(OversightHook):
    """Stop a run whose turn has closed, before it spends another model call."""

    allowed_hook_points = ["before_llm"]  # noqa: RUF012

    def run(self, state: State) -> None:
        if outlived_turn():
            state.set("stop_run", "turn closed")

    async def run_async(self, state: State) -> None:
        self.run(state)


def request_of(context: dict[str, Any] | None) -> TurnRequest | None:
    """The TurnRequest in a hook_context, if it has one."""
    return (context or {}).get(REQUEST_KEY)


def turn_of(context: dict[str, Any] | None) -> Turn | None:
    """The turn a run with this hook_context belongs to, once it started."""
    request = request_of(context)
    return request.turn if request is not None else None


def request_in(state: State) -> TurnRequest:
    """The run's TurnRequest. A run started without one gets the defaults."""
    context = state.data.get("hook_context")
    if context is None:
        context = {}
        state.set("hook_context", context)
    return context.setdefault(REQUEST_KEY, TurnRequest())


async def close_turn(context: dict[str, Any] | None, *, error: str = "") -> None:
    """Close the run's own turn, if it is still open, recording `error`."""
    turn = turn_of(context)
    if turn is not None and turn.owns:
        await turn.guard.close(error=error)


async def run_agent(
    agent: Any, messages: list[Any], *, hook_context: dict[str, Any] | None = None, **kwargs: Any
) -> dict[str, Any]:
    """Run a Haystack Agent with oversight, and close its turn even when the run fails.

    The agent gets the oversight hooks and tools here (see `install`), so every
    run through this function is checked. The run gets its own task, so the turn
    it makes current ends with it. Read the turn with `turn_of(hook_context)`.
    """
    from django_ai_sdk.oversight.plugin import install

    context = {} if hook_context is None else hook_context
    run = install(agent).run_async(messages=messages, hook_context=context, **kwargs)
    try:
        return await asyncio.create_task(run)
    except BaseException as exc:
        await close_turn(context, error=_run_error(exc))
        raise


def _question(state: State) -> str:
    for message in reversed(state.data.get("messages") or []):
        if message.is_from("user"):
            return message.text or ""
    return ""


def _leave(turn: Turn | None) -> None:
    if turn is not None and turn.token is not None:
        leave(turn.token)
        turn.token = None


def _run_error(exc: BaseException) -> str:
    if isinstance(exc, GeneratorExit | asyncio.CancelledError):
        return "disconnected"
    return type(exc).__name__


__all__ = [
    "REQUEST_KEY",
    "OversightHook",
    "OversightStopHook",
    "Turn",
    "TurnCloseHook",
    "TurnOpenHook",
    "TurnRequest",
    "close_turn",
    "request_in",
    "request_of",
    "run_agent",
    "turn_of",
]
