"""A turn around model calls of your own, outside a Haystack Agent run.

    async with turn(agent, channel="workflow:digest:write", user=user, input=text) as t:
        question = await t.gate()  # raises Blocked
        answer = await my_model(question)
        return await t.finish(answer)  # masked; raises Blocked

It opens, joins or skips a turn as `run_with_oversight` does, and inside it the SDK's
generators count as overseen.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from django_ai_sdk.common import ChatMessage, last_user_message, last_user_text
from django_ai_sdk.errors import Blocked
from django_ai_sdk.oversight.calls import settled
from django_ai_sdk.oversight.guard import TurnGuard, closing_turn, current_guard, guarding
from django_ai_sdk.oversight.plugin import open_turn, settled_outside_a_turn
from django_ai_sdk.oversight.plugin.hooks import JOINED, withhold_unfit
from django_ai_sdk.oversight.types import Stage

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.oversight.policy import Policy


class OpenTurn:
    """The turn that `turn()` yields.

    When no turn opened (no policy, oversight off, or the run joined an outer
    turn), `guard` is None and `gate()` and `finish()` return their input; the
    outer turn judges a joined question like a tool result.
    Otherwise use `guard` to report what the run found (`set_sources`, `add_usage`).
    """

    def __init__(
        self,
        guard: TurnGuard | None,
        *,
        messages: list[ChatMessage] | None,
        input: str | None,
    ) -> None:
        self.guard = guard
        # The history and question to send the model, masked by the gate.
        self.messages: list[ChatMessage] = list(messages or [])
        self.input = input if input is not None else last_user_text(self.messages)
        self._gated = False

    async def gate(self) -> str:
        """Judge the history and the question before the model reads them.

        Returns the question, masked if a rule said so. `messages` then holds the
        history and question to send. Raises `Blocked`.
        """
        guard = self.guard
        if self._gated:
            return self.input
        self._gated = True
        if guard is None:
            if (outer := current_guard()) is not None and self.input:
                # Joined: no gate of its own, so the outer turn judges it like a tool result.
                masked = await outer.mask_result(JOINED, self.input)
                if masked != self.input and (asked := last_user_message(self.messages)):
                    asked.content = masked
                self.input = masked
            return self.input
        if self.messages:
            # Copies, but the question stays shared so masking reaches the caller.
            question = last_user_message(self.messages)
            self.messages = [m if m is question else m.model_copy(deep=True) for m in self.messages]
            # The question last: every other message, a reply after it too, is history.
            history = [m for m in self.messages if m is not question]
            asked = question or ChatMessage(role="user", content="")
            checked = await guard.gate_history([*history, asked])
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
        """The answer as the policy allows it; raises `Blocked`.

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
        # As in Agent.run: this block opts out, without turning off agents run inside it.
        with guarding(None), settled():
            yield OpenTurn(None, messages=messages, input=input)
        return
    guard, opened = await open_turn(
        agent,
        channel=channel,
        user=user,
        messages=messages,
        input=input,
        oversight=oversight,
    )
    current = OpenTurn(guard if opened else None, messages=messages, input=input)
    if not opened:
        with settled_outside_a_turn():
            yield current
        return
    async with closing_turn(guard):
        # Not settled: a model call counts as overseen once this turn's gate has run.
        with guarding(guard):
            yield current


def _blocked(guard: TurnGuard, stage: Stage, message: str) -> Blocked:
    return Blocked(message, stage=str(stage), turn_id=guard.id)


__all__ = ["OpenTurn", "turn"]
