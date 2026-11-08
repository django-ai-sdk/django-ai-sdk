"""TurnGuard checks one turn, whatever channel it came from.

Chat, workflow steps and your own code open a guard and report what happens
(sources found, tools called, text produced). The guard is asked whether to go on
three times: at the gate, before each tool call, and at finish. Closing it writes
the turn to the record (see `persist`) and queues the background stages.

    guard = await TurnGuard.start(agent, channel="api", user=user, input=text)
    gate = await guard.gate()
    if gate.blocked:
        return gate.message
    answer = await run_my_agent(gate.input)
    finished = await guard.finish(answer)
    await guard.close()
    return finished.output

A guard for an agent without a policy, or with oversight off, is inert. Every
call returns at once and allows everything.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
from typing import TYPE_CHECKING, Any, Literal

from asgiref.sync import sync_to_async
from django.utils import timezone

from django_ai_sdk.errors import Blocked
from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.agents import acode_policy, agent_key, fingerprint
from django_ai_sdk.oversight.conf import is_enabled, setting
from django_ai_sdk.oversight.guard.finish import FinishResult, FinishStage
from django_ai_sdk.oversight.guard.gate import GateResult, GateStage
from django_ai_sdk.oversight.guard.notices import (
    TOO_LONG_MESSAGE,
    TOOL_UNAVAILABLE,
    UNCHECKED_TEXT,
    rule_notes,
)
from django_ai_sdk.oversight.guard.tool import ToolStage
from django_ai_sdk.oversight.persist import persist_turn
from django_ai_sdk.oversight.types import Capture, TurnContext
from django_ai_sdk.utils import scoped

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.oversight.policy import Policy, ResolvedPolicy

logger = get_logger(__name__)

_unguarded: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "django_ai_sdk_oversight_unguarded", default=False
)
_current: contextvars.ContextVar[TurnGuard | None] = contextvars.ContextVar(
    "django_ai_sdk_oversight_guard", default=None
)


def unguarded() -> contextlib.AbstractContextManager[None]:
    """Run code that must not open turns of its own, such as judges and titles."""
    return scoped(_unguarded, True)


def is_unguarded() -> bool:
    """True inside `unguarded()`."""
    return _unguarded.get()


def guarding(guard: TurnGuard | None) -> contextlib.AbstractContextManager[None]:
    """Make `guard` the current guard for tool hooks and get_tools(). Tasks inherit it."""
    return scoped(_current, guard if guard is not None and guard.active else None)


@contextlib.asynccontextmanager
async def closing_turn(
    guard: TurnGuard, *, keep_open: bool = False, error: Callable[[], str] = str
) -> AsyncGenerator[None]:
    """Close `guard` when the block ends, recording an exception as the run's error.

    A client that left is recorded as "disconnected". `Blocked` is not an error,
    since the refusal is already on the turn. `error` gives the error to record
    otherwise, read at close. With `keep_open` the guard is closed only when the
    block raises, because something else (the Stream) takes the turn over.
    """
    try:
        yield
    except BaseException as exc:
        await guard.close(error=_run_error(exc) or error())
        raise
    if not keep_open:
        await guard.close(error=error())


def _run_error(exc: BaseException) -> str:
    if isinstance(exc, Blocked):
        return ""
    if isinstance(exc, GeneratorExit | asyncio.CancelledError):
        return "disconnected"
    return type(exc).__name__


def current_guard() -> TurnGuard | None:
    """Return the open guard this code runs under, or None.

    A task started inside a turn inherits its guard and may outlive it, so a
    closed guard counts as none.
    """
    guard = _current.get()
    return guard if guard is not None and not guard._closed else None


class TurnGuard(GateStage, ToolStage, FinishStage):
    """One turn under a policy. Build it with `start()`. Inert when `active` is False.

    The stages run in order: `gate` (gate.py), `tool_call` (tool.py), `finish`
    (finish.py), then `close` here. Shared state lives in GuardState (state.py).
    Asking lenses and matching rules lives in asking.py.
    """

    # Opening.

    @classmethod
    async def start(
        cls,
        agent: Any,
        *,
        channel: str,
        user: AbstractBaseUser | AnonymousUser | None = None,
        thread_id: str | None = None,
        messages: list[ChatMessage] | None = None,
        input: str | None = None,
        oversight: type[Policy] | str | Literal[False] | None = None,
    ) -> TurnGuard:
        """Open a guard for one turn of `agent`. Inert unless a policy applies.

        `oversight` overrides the agent's policy for this turn. It can be a Policy
        subclass or a name. False or "" (no policy) opens no turn. None uses the
        overlay's choice (`OverlaySource.agent_policies`), falling back to the
        agent's code (`acode_policy`). An agent with `oversight_required` ignores an overlay
        that switches oversight off.

        If the policy cannot be chosen (`get_oversight` raised) or loaded (unknown
        or broken), every stage refuses with Block.DEFAULT_MESSAGE and the turn is
        recorded as blocked with the error. With FAIL_MODE "open" the turn runs
        unguarded instead, and the failure is logged.
        """
        if oversight in (False, "") or is_unguarded() or not is_enabled():
            return cls()
        from django_ai_sdk.oversight.registry import policy_name
        from django_ai_sdk.oversight.resolve import aassigned_policy, aget_policy

        opening = {
            "agent": agent,
            "channel": channel,
            "user": user,
            "thread_id": thread_id,
            "messages": messages,
            "input": input,
        }
        name = ""
        try:
            name = policy_name(oversight) if oversight else ""
            if not name:
                # The overlay's choice wins over the agent's code.
                assigned = await aassigned_policy(agent_key(agent))
                if assigned == "" and getattr(agent, "oversight_required", False):
                    # The agent's code requires oversight, so the overlay cannot turn it off.
                    logger.warning(
                        "Oversight ignores an overlay switching off {!r}: it requires oversight",
                        agent_key(agent),
                    )
                    assigned = None
                if assigned is None:
                    name = (await acode_policy(agent, channel=channel, user=user))[0]
                else:
                    name = assigned
        except Exception:
            logger.exception("Oversight could not choose a policy for {!r}", agent_key(agent))
            return cls._unloadable(name="", error="policy could not be chosen", **opening)
        if not name:
            return cls()
        try:
            resolved = await aget_policy(name)
        except Exception:
            logger.exception("Oversight policy {!r} could not be loaded", name)
            resolved = None
        if resolved is None:
            return cls._unloadable(name=name, error="policy could not be loaded", **opening)
        return cls._opened(resolved, **opening)

    @classmethod
    def _opened(
        cls,
        policy: ResolvedPolicy,
        *,
        agent: Any,
        channel: str,
        user: Any,
        thread_id: str | None,
        messages: list[ChatMessage] | None,
        input: str | None,
    ) -> TurnGuard:
        messages = messages or []
        if input is None:
            input = next((m.content for m in reversed(messages) if m.role == "user"), "") or ""
        guard = cls(
            active=True,
            policy=policy,
            channel=channel,
            thread_id=str(thread_id or ""),
            actor_id=_actor_id(user),
            created_at=timezone.now(),
            original_input=input,
        )
        guard.context = TurnContext(
            turn_id=guard.id,
            channel=channel,
            agent_id=agent_key(agent),
            agent_name=str(getattr(agent, "name", "") or type(agent).__name__),
            input=input,
        )
        guard.set_history(messages)
        guard.agent_fingerprint = fingerprint(agent)
        return guard

    @classmethod
    def _unloadable(cls, *, name: str, error: str, **opening: Any) -> TurnGuard:
        """Build the guard for a turn whose policy could not be chosen or loaded.

        It refuses everything under an empty stand-in policy, so the refusal and
        the error are still recorded. With FAIL_MODE "open" it is inert instead.
        """
        from django_ai_sdk.oversight.policy import default_fail_mode

        if default_fail_mode() == "open":
            logger.error(
                "Oversight runs a turn of {!r} unguarded: its {} (FAIL_MODE is open)",
                agent_key(opening["agent"]),
                error,
            )
            return cls()
        try:
            guard = cls._opened(_stand_in(name), **opening)
        except Exception:
            # The refusal cannot be recorded, so refuse without a record.
            logger.exception("Oversight could not open a refusing turn")
            guard = cls(active=True, policy=_stand_in(name), _closed=True)
        guard._unloaded = True
        guard.refused = True
        guard.note_error(error)
        logger.error(
            "Oversight refuses a turn of {!r}: its {}",
            agent_key(opening["agent"]),
            error,
        )
        return guard

    # Closing.

    async def close(self, *, error: str = "") -> str | None:
        """Write the turn to the record and queue the background stages. Never raises.

        Tool results still being masked are cancelled first, so nothing is added
        to a closed turn. `error` is what went wrong in the run itself, such as the
        caller's exception or a disconnect. It is recorded with oversight's own
        errors. If set, a turn no rule acted on gets a blank outcome.

        Returns the turn id, or None for an inert guard or a failed write.
        """
        if not self.active or self._closed:
            return None
        self._closed = True
        await self._cancel_masking()
        self.note_error(error)
        self.run_error = error
        if not self.recording:
            return None
        try:
            turn_id = await sync_to_async(persist_turn)(self)
        except Exception:
            # Log the outcome and errors, never the content.
            logger.exception(
                "Oversight could not record turn {}: outcome {}, acted {}, errors {}",
                self.id,
                self.outcome,
                sorted({m.rule.name for m in self.matched if m.acted}),
                self.error or "none",
            )
            return None
        return turn_id


def _stand_in(name: str) -> ResolvedPolicy:
    """Return an empty policy named `name`, used to refuse a turn whose policy failed to load."""
    from django_ai_sdk.oversight.conf import validate_retention
    from django_ai_sdk.oversight.policy import ResolvedPolicy

    return ResolvedPolicy(
        name=name,
        capture=Capture(setting("CAPTURE")),
        retention=validate_retention(setting("RETENTION")),
        origin="unloaded",
    )


def _actor_id(user: Any) -> str:
    """Return the record's id for `user` (custody's `actor`). "" when anonymous or missing."""
    from django_ai_sdk.oversight import custody

    pk = getattr(user, "pk", None)
    return "" if pk is None else custody.get().actor(str(pk))


__all__ = [
    "TOOL_UNAVAILABLE",
    "TOO_LONG_MESSAGE",
    "UNCHECKED_TEXT",
    "FinishResult",
    "GateResult",
    "TurnGuard",
    "closing_turn",
    "current_guard",
    "guarding",
    "is_unguarded",
    "rule_notes",
    "unguarded",
]
