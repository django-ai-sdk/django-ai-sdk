"""A turn around model calls of your own, outside a Haystack Agent run.

    async with turn(agent, channel="workflow:digest:write", user=user, input=text) as t:
        question = await t.gate()  # raises Blocked
        answer = await my_model(question)
        return await t.finish(answer)  # masked; raises Blocked

It opens, joins or skips a turn as `run_with_oversight` does (`oversight.plugin`),
and inside it the SDK's generators count as overseen (`oversight.calls`).
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from django_ai_sdk.common import last_user_message, last_user_text
from django_ai_sdk.errors import Blocked
from django_ai_sdk.oversight.calls import settled
from django_ai_sdk.oversight.guard import TurnGuard, closing_turn, guarding, unguarded
from django_ai_sdk.oversight.history import thread_history, thread_storage
from django_ai_sdk.oversight.plugin import open_turn
from django_ai_sdk.oversight.plugin.hooks import withhold_unfit
from django_ai_sdk.oversight.types import Stage

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.oversight.policy import Policy


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
            # A history the stored thread vouched for is not judged again.
            checked = await guard.gate_history(self.messages, history=vouched is None)
            if checked.blocked:
                raise _blocked(guard, Stage.GATE, checked.message)
        gate = await guard.gate()
        if gate.blocked:
            raise _blocked(guard, Stage.GATE, gate.message)
        if gate.changed and (question := last_user_message(self.messages)) is not None:
            question.content = gate.text
        self.input = gate.text
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
        finished = await guard.finish(result.model_dump_json() if schema else str(result))
        if schema is not None:
            withhold_unfit(guard, finished, schema)
        if finished.blocked:
            raise _blocked(guard, Stage.FINISH, finished.message)
        if not finished.changed:
            return result
        return schema.model_validate_json(finished.text) if schema else finished.text


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
        with settled():
            yield current
        return
    async with closing_turn(guard, error=lambda: current.error):
        with guarding(guard), settled():
            yield current


def _working_copy(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Copy every message but the last user message, which stays shared so masking reaches the caller."""
    question = last_user_message(messages)
    return [m if m is question else m.model_copy(deep=True) for m in messages]


def _blocked(guard: TurnGuard, stage: Stage, message: str) -> Blocked:
    return Blocked(message, stage=str(stage), turn_id=guard.id)


__all__ = ["OpenTurn", "turn"]
