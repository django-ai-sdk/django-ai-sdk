"""Opening, gating and finishing a turn of an agent.

The plugin's hooks (see `oversight.plugin`) run these steps inside a Haystack
Agent run. `turn()` runs the same steps around model calls of your own.

A turn inside another one joins it when its policy is the same (or there is
none). Another policy opens a child turn, linked to the outer one.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Literal

from django.core.exceptions import ImproperlyConfigured
from pydantic import BaseModel, ValidationError

from django_ai_sdk.common import last_user_message, last_user_text
from django_ai_sdk.errors import Blocked
from django_ai_sdk.oversight.actions import Block
from django_ai_sdk.oversight.guard import (
    TurnGuard,
    closing_turn,
    current_guard,
    guarding,
    unguarded,
)
from django_ai_sdk.oversight.guard.state import merge_restrict
from django_ai_sdk.oversight.history import thread_history, thread_storage
from django_ai_sdk.oversight.types import Stage

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.oversight.guard import FinishResult, GateResult
    from django_ai_sdk.oversight.policy import Policy


def validate_oversight(cls: type) -> None:
    """Check `cls.oversight` is a Policy subclass, a policy name, False or None.

    Any other value would silently leave the agent unguarded, so fail at import.
    Registering a Policy subclass here lets the system checks see it.
    """
    value = cls.__dict__.get("oversight")
    if value is None or value is False or (isinstance(value, str) and value):
        return
    if isinstance(value, type):
        from django_ai_sdk.oversight.policy import Policy

        if issubclass(value, Policy):
            from django_ai_sdk.oversight.registry import policy_name

            policy_name(value)
            return
    raise ImproperlyConfigured(
        f"{cls.__qualname__}.oversight must be a Policy subclass, a policy's name, False "
        f"or None, not {value!r}."
    )


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


async def gate_conversation(
    guard: TurnGuard, messages: list[ChatMessage], *, vouched: bool
) -> GateResult:
    """Judge the history and the question, masking each in `messages` in place.

    A history the stored thread `vouched` for is not judged again, but attachment
    context always is: it is rebuilt from the document each turn and the model reads it.
    """
    if messages:
        checked = await guard.gate_history(messages, history=not vouched)
        if checked.blocked:
            return checked
    gate = await guard.gate()
    if gate.masked and (question := last_user_message(messages)) is not None:
        question.content = gate.input
    return gate


def withhold_unfit(guard: TurnGuard, finished: FinishResult, schema: type[BaseModel]) -> None:
    """Withhold a masked structured answer whose JSON no longer fits `schema`.

    Returning the unmasked original instead would leak what the mask hid.
    """
    if not finished.replaced or finished.blocked:
        return
    try:
        schema.model_validate_json(finished.output)
    except ValidationError:
        guard.refuse("masked result does not fit its schema")
        finished.blocked = True
        finished.message = finished.output = str(Block.DEFAULT_MESSAGE)


class OpenTurn:
    """The turn that `turn()` yields. Gate the question, then finish the answer.

    When no turn opened (no policy, oversight off, or the run joined an outer
    turn), `guard` is None and `gate()` and `finish()` return their input.
    Otherwise use `guard` to report what the run found (`set_sources`, `add_usage`).
    """

    def __init__(
        self,
        agent: Any,
        guard: TurnGuard | None,
        *,
        thread_id: str | None,
        messages: list[ChatMessage] | None,
        input: str | None,
    ) -> None:
        self.agent = agent
        self.guard = guard
        self.thread_id = thread_id
        # The history and question to send the model, masked by the gate.
        self.messages: list[ChatMessage] = list(messages or [])
        self.input = input if input is not None else last_user_text(self.messages)
        # An error that did not raise, such as "no result". Recorded at close.
        self.error = ""
        self._gated = False

    @property
    def id(self) -> str:
        """The turn's id, or "" when no turn opened."""
        return self.guard.id if self.guard is not None else ""

    async def gate(self) -> str:
        """Judge the history and the question before the model reads them.

        Returns the question, masked if a rule said so. `messages` then holds the
        history and question to send. Raises `Blocked`.
        """
        if self._gated:
            return self.input
        self._gated = True
        guard = self.guard
        if guard is None:
            return self.input
        vouched = None
        if self.messages:
            self.messages = _working_copy(self.messages)
            storage = await thread_storage(self.agent, self.thread_id)
            vouched = (
                await thread_history(guard.resolved, storage, self.messages) if storage else None
            )
            if vouched is not None:
                self.messages = vouched
        gate = await gate_conversation(guard, self.messages, vouched=vouched is not None)
        if gate.blocked:
            raise _blocked(guard, Stage.GATE, gate.message)
        self.input = gate.input
        return self.input

    async def finish(self, result: Any) -> Any:
        """Return the answer as the policy allows it, or raise `Blocked`.

        A Pydantic model comes back as the same model, rebuilt from its masked
        JSON. Anything else comes back as its masked text.
        """
        guard = self.guard
        if guard is None:
            return result
        if not self._gated:
            await self.gate()
        schema = type(result) if isinstance(result, BaseModel) else None
        text = result.model_dump_json() if schema else str(result)
        finished = await guard.finish(text)
        if schema is not None:
            withhold_unfit(guard, finished, schema)
        if finished.blocked:
            raise _blocked(guard, Stage.FINISH, finished.message)
        if not finished.replaced:
            return result
        return schema.model_validate_json(finished.output) if schema else finished.output


@contextlib.asynccontextmanager
async def turn(
    agent: Any,
    *,
    channel: str,
    user: AbstractBaseUser | AnonymousUser | None = None,
    thread_id: str | None = None,
    messages: list[ChatMessage] | None = None,
    input: str | None = None,
    oversight: type[Policy] | str | Literal[False] | None = None,
) -> AsyncGenerator[OpenTurn]:
    """Open a turn of `agent` around your own model calls. It closes when the block ends.

        async with turn(agent, channel="workflow:digest:write", user=user, input=text) as t:
            question = await t.gate()  # raises Blocked
            answer = await my_model(question)
            return await t.finish(answer)  # masked; raises Blocked

    Pass `messages` (history, then the question) or `input` (just the question).
    `oversight` works as in `Agent.run`. Inside the block, tool hooks see the turn.
    An exception other than `Blocked` is recorded as the run's error.
    """
    if oversight is False:
        with guarding(None), unguarded():
            yield OpenTurn(agent, None, thread_id=thread_id, messages=messages, input=input)
        return
    guard, opened = await open_turn(
        agent,
        channel=channel,
        user=user,
        thread_id=thread_id,
        messages=messages,
        input=input,
        oversight=oversight,
    )
    current = OpenTurn(
        agent, guard if opened else None, thread_id=thread_id, messages=messages, input=input
    )
    if not opened:
        # Joined the outer turn, or no policy: an opt-out on purpose.
        yield current
        return
    async with closing_turn(guard, error=lambda: current.error):
        with guarding(guard):
            yield current


def _working_copy(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Copy every message except the last user message.

    That message stays shared so masking it also masks the caller's copy. The
    rest of the caller's list is left alone.
    """
    question = last_user_message(messages)
    return [m if m is question else m.model_copy(deep=True) for m in messages]


def _blocked(guard: TurnGuard, stage: Stage, message: str) -> Blocked:
    return Blocked(message, stage=str(stage), turn_id=guard.id)


__all__ = [
    "OpenTurn",
    "gate_conversation",
    "joins_turn",
    "open_turn",
    "turn",
    "validate_oversight",
    "withhold_unfit",
]
