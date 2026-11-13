"""The turn, carried by the run: opened in before_run, closed in after_run.

The run's `hook_context` names the turn (`channel`, `user`, `thread_id`,
`oversight`) and gets it back under `TURN_KEY`, so the caller can read what
oversight decided once the run returns.

Haystack skips after_run when a run raises or is cancelled. `run_agent` closes
the turn then, with the error. A run started without it leaves its turn open,
written ahead after the gate, until the sweep marks it interrupted.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from haystack.core.serialization import default_from_dict, default_to_dict

from django_ai_sdk.errors import Blocked
from django_ai_sdk.oversight.agent_run import joins_turn
from django_ai_sdk.oversight.guard import (
    TurnGuard,
    current_guard,
    enter,
    leave,
    outlived_turn,
)
from django_ai_sdk.oversight.guard.state import merge_restrict
from django_ai_sdk.oversight.types import Stage

if TYPE_CHECKING:
    import contextvars

    from haystack.components.agents.state.state import State

    from django_ai_sdk.oversight.guard import FinishResult, GateResult

# Where the run's hook_context holds its Turn.
TURN_KEY = "django_ai_sdk.oversight.turn"


@dataclass
class Turn:
    """The turn a run belongs to. `owns` is False when it joined an outer turn or has no policy."""

    guard: TurnGuard
    owns: bool
    gated: GateResult | None = None
    # Index of the first message the run produced, after the gate.
    first: int = 0
    token: contextvars.Token[TurnGuard | None] | None = None

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


class OversightHook:
    """Base for the oversight hooks: serializable, so an Agent with them still saves to YAML."""

    allowed_hook_points: list[str] = []  # noqa: RUF012

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OversightHook:
        return default_from_dict(cls, data)


class TurnOpenHook(OversightHook):
    """Open the run's turn under `agent`'s policy, or join the turn it runs inside.

    A run inside another turn joins it when the policy is the same (or there is
    none). A different policy opens a child turn, linked to the outer one, that
    keeps its person and the tools it took away.
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
        _context(state)[TURN_KEY] = Turn(current_guard() or TurnGuard(), owns=False)

    async def run_async(self, state: State) -> None:
        context = _context(state)
        agent = self.agent or context.get("agent")
        outer = current_guard()
        if agent is None:
            context[TURN_KEY] = Turn(outer or TurnGuard(), owns=False)
            return
        guard = await TurnGuard.start(
            agent,
            channel=context.get("channel") or "run",
            user=context.get("user"),
            thread_id=context.get("thread_id"),
            input=_question(state),
            oversight=context.get("oversight"),
        )
        if not guard.active or joins_turn(outer, guard.resolved.name):
            context[TURN_KEY] = Turn(outer or guard, owns=False)
            return
        if outer is not None:
            guard.parent_turn_id = outer.id
            guard.actor_id = guard.actor_id or outer.actor_id
            if outer.restrict is not None:
                guard.restrict = merge_restrict(guard.restrict, outer.restrict)
        guard.message_id = context.get("message_id") or guard.message_id
        guard.on_tool_results = context.get("on_tool_results")
        context[TURN_KEY] = Turn(guard, owns=True, token=enter(guard))


class TurnCloseHook(OversightHook):
    """Close and record the run's turn, last thing in the run.

    With `keep_open` in the hook_context the caller closes it instead, such as a
    stream that still stores the answer the turn points at.
    """

    allowed_hook_points = ["after_run"]  # noqa: RUF012

    def run(self, state: State) -> None:
        _leave(turn_of(_context(state)))

    async def run_async(self, state: State) -> None:
        context = _context(state)
        turn = turn_of(context)
        _leave(turn)
        if turn is not None and turn.owns and not context.get("keep_open"):
            await turn.guard.close()


class OversightStopHook(OversightHook):
    """Stop a run whose turn has closed, before it spends another model call."""

    allowed_hook_points = ["before_llm"]  # noqa: RUF012

    def run(self, state: State) -> None:
        if outlived_turn():
            state.set("stop_run", "turn closed")

    async def run_async(self, state: State) -> None:
        self.run(state)


def turn_of(context: dict[str, Any] | None) -> Turn | None:
    """The turn a run with this hook_context belongs to, once it started."""
    return (context or {}).get(TURN_KEY)


async def close_turn(context: dict[str, Any] | None, *, error: str = "") -> None:
    """Close the run's own turn, if it is still open, recording `error`."""
    turn = turn_of(context)
    if turn is not None and turn.owns:
        await turn.guard.close(error=error)


async def run_agent(
    agent: Any, messages: list[Any], *, hook_context: dict[str, Any] | None = None, **kwargs: Any
) -> dict[str, Any]:
    """Run a Haystack Agent with oversight, and close its turn even when the run fails.

    The run gets its own task, so the turn it makes current ends with it. The
    turn is in `hook_context[TURN_KEY]` afterwards.
    """
    from django_ai_sdk.oversight.plugin import install

    context = {} if hook_context is None else hook_context
    run = install(agent, agent=context.get("agent")).run_async(
        messages=messages, hook_context=context, **kwargs
    )
    try:
        return await asyncio.create_task(run)
    except BaseException as exc:
        await close_turn(context, error=_run_error(exc))
        raise


def _context(state: State) -> dict[str, Any]:
    context = state.data.get("hook_context")
    if context is None:
        context = {}
        state.set("hook_context", context)
    return context


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
    "TURN_KEY",
    "OversightHook",
    "OversightStopHook",
    "Turn",
    "TurnCloseHook",
    "TurnOpenHook",
    "close_turn",
    "run_agent",
    "turn_of",
]
