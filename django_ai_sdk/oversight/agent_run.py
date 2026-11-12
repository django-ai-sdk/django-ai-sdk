"""Run an agent outside chat as one oversight turn. `Agent.run()` uses this.

Chat goes through the Stream adapter. Every other way to run an agent (the HTTP
run endpoints, workflow steps, your own code) goes through `run_turn`. That way
no entry point skips the policy.

Each agent runs under its own policy. A run inside another turn joins that turn
when it resolves the same policy or no policy. With a different policy it opens
a child turn, with its own gate, tool checks, finish and record, linked to the
outer turn.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import weakref
from typing import TYPE_CHECKING, Any, Literal

from django.core.exceptions import ImproperlyConfigured
from pydantic import BaseModel, ValidationError

from django_ai_sdk.common import as_chat_messages, last_user_message, last_user_text
from django_ai_sdk.errors import Blocked
from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.actions import Block
from django_ai_sdk.oversight.guard import (
    TurnGuard,
    closing_turn,
    current_guard,
    guarding,
    unguarded,
)
from django_ai_sdk.oversight.guard.masks import gate_masking_lenses
from django_ai_sdk.oversight.guard.state import merge_restrict
from django_ai_sdk.oversight.types import Stage
from django_ai_sdk.tracing import bind as bind_trace

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.oversight.guard import GateResult
    from django_ai_sdk.oversight.policy import Policy
    from django_ai_sdk.storage.base import BaseStorageAdapter

logger = get_logger(__name__)


async def run_turn(
    agent: Any,
    work: Callable[[list[ChatMessage]], Awaitable[Any]],
    messages: list[ChatMessage],
    *,
    channel: str,
    user: AbstractBaseUser | AnonymousUser | None = None,
    thread_id: str | None = None,
    oversight: type[Policy] | str | Literal[False] | None = None,
) -> Any:
    """Run `work(messages)` as one turn of `agent`.

    `oversight` picks the policy:
    - None uses the agent's own, from its code.
    - A Policy subclass or a policy name replaces it.
    - "" means the agent has no policy of its own.
    - False runs the work unguarded, even inside another turn.

    The history is judged before the question, unless a stored thread vouches
    for it (see `thread_history`). The caller's list is not rewritten. History
    is judged on copies. Only the last user message is masked in place, so the
    caller stores the masked question.

    Tool calls are checked while the work runs, and the answer is finished
    before it is returned. Raises `Blocked` when the gate or the finish stops it.
    """
    # Bind the thread so headless runs are traced to it.
    with bind_trace(thread_id=thread_id):
        async with turn(
            agent,
            channel=channel,
            user=user,
            thread_id=thread_id,
            messages=messages,
            oversight=oversight,
        ) as current:
            await current.gate()
            result = await work(current.messages)
            if result is None:
                current.error = "no result"
                return None
            return await current.finish(result)


# run() overrides that are already wrapped, so a subclass is not wrapped twice.
_guarded_runs: weakref.WeakSet[Callable[..., Any]] = weakref.WeakSet()


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


def guard_run_override(cls: type) -> None:
    """Wrap `cls`'s `run()` in a turn, so overriding it cannot skip the policy.

    A `run()` from a mixin counts too. The turn opens around the outermost `run()`.
    When the override calls `super().run()`, that call resolves the same policy and
    joins the turn.
    """
    from django_ai_sdk.agents.base import Agent

    run = getattr(cls, "run", None)
    if not callable(run) or run is Agent.run or run in _guarded_runs:
        return

    @functools.wraps(run)
    async def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
        call = _run_arguments(args, kwargs)
        messages = call.get("messages")
        if messages is None:
            # Nothing to judge yet. The base run() opens the turn when it is called.
            return await run(self, *args, **kwargs)

        async def work(given: list[ChatMessage]) -> Any:
            if "messages" in kwargs:
                return await run(self, *args, **{**kwargs, "messages": given})
            return await run(self, given, *args[1:], **kwargs)

        # Give the override ChatMessages, like the base run() would.
        return await run_turn(
            self,
            work,
            as_chat_messages(messages),
            channel=call.get("channel", "run"),
            user=call.get("user"),
            thread_id=call.get("thread_id"),
            oversight=call.get("oversight"),
        )

    _guarded_runs.add(guarded)
    cls.run = guarded  # ty: ignore[unresolved-attribute]


@functools.cache
def _run_signature() -> inspect.Signature:
    """`Agent.run`'s signature, to read an override's call arguments by name."""
    from django_ai_sdk.agents.base import Agent

    return inspect.signature(Agent.run)


def _run_arguments(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Name an override's call arguments as `Agent.run` would."""
    signature = _run_signature()
    try:
        return signature.bind_partial(None, *args, **kwargs).arguments
    except TypeError:  # the override takes arguments of its own: read only the base ones
        params = signature.parameters
        names = [n for n, p in params.items() if p.kind is p.POSITIONAL_OR_KEYWORD]
        return {
            **dict(zip(names, (None, *args))),
            **{k: v for k, v in kwargs.items() if k in params},
        }


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
            storage = await _thread_storage(self.agent, self.thread_id)
            vouched = await thread_history(guard, storage, self.messages) if storage else None
            if vouched is not None:
                self.messages = vouched
        gate = await gate_conversation(guard, self.messages, vouched=vouched is not None)
        if gate.blocked:
            raise _blocked(guard, Stage.GATE, gate.message)
        self.input = gate.input
        return self.input

    async def finish(self, result: Any) -> Any:
        """Return the answer as the policy allows it, or raise `Blocked`.

        A string comes back as a string. A Pydantic model comes back as the same
        model, rebuilt from its masked JSON.
        """
        if self.guard is None:
            return result
        if not self._gated:
            await self.gate()
        return await _finish(self.guard, result)


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
    `oversight` works as in `run_turn`. Inside the block, tool hooks see the turn.
    An exception other than `Blocked` is recorded as the run's error.
    """
    if oversight is False:
        with guarding(None), unguarded():
            yield OpenTurn(agent, None, thread_id=thread_id, messages=messages, input=input)
        return
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
    # Inside another turn, the same policy (or none) joins it; another opens a child turn.
    opened = guard.active and not joins_turn(outer, guard.resolved.name)
    if opened and outer is not None:
        guard.parent_turn_id = outer.id
        # A child turn is still the outer turn's person, and keeps the tools it took away.
        guard.actor_id = guard.actor_id or outer.actor_id
        if outer.restrict is not None:
            guard.restrict = merge_restrict(guard.restrict, outer.restrict)
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


async def gate_conversation(
    guard: TurnGuard, messages: list[ChatMessage], *, vouched: bool
) -> GateResult:
    """Judge the history and the question, then mask the question in `messages` in place.

    Chat and `OpenTurn.gate` both use this. A history the stored thread `vouched`
    for is not judged again, but attachment context always is: it is rebuilt from
    the document each turn and the model reads it.
    """
    if messages:
        checked = await guard.gate_history(messages, history=not vouched)
        if checked.blocked:
            return checked
    gate = await guard.gate()
    if gate.masked and (question := last_user_message(messages)) is not None:
        question.content = gate.input
    return gate


def joins_turn(outer: TurnGuard | None, policy_name: str) -> bool:
    """Whether a run under `policy_name` joins the open turn `outer` instead of its own."""
    return outer is not None and outer.policy is not None and outer.policy.name == policy_name


def takes_stored_history(guard: TurnGuard) -> bool:
    """Whether the turn uses the stored thread as its history.

    Only when the policy enforces masking at the gate. Then the stored copy is
    the masked one, and each message in it was judged in its own turn. Otherwise
    the client's history is used and judged by `gate_history`, so a policy in
    report mode changes nothing.
    """
    return guard.active and bool(gate_masking_lenses(guard.resolved))


async def thread_history(
    guard: TurnGuard, storage_adapter: BaseStorageAdapter, messages: list[ChatMessage]
) -> list[ChatMessage] | None:
    """Return the history the stored thread vouches for, or None if it must be judged.

    Under a policy that masks at the gate, the stored messages replace the sent
    ones (see `stored_history`). Otherwise a sent history that matches the stored
    thread (role and text) was already judged and is kept. One that differs was
    edited or made up, so it gets judged. Chat and the run endpoints both use this.
    """
    if takes_stored_history(guard):
        return await stored_history(storage_adapter, messages)
    if await _as_stored(storage_adapter, messages):
        return messages
    return None


async def stored_history(
    storage_adapter: BaseStorageAdapter, messages: list[ChatMessage]
) -> list[ChatMessage]:
    """Replace the history in `messages` with the stored thread. Keep the new question.

    The client's copy may be unmasked or made up, so it is not trusted. When every
    earlier message has a stored id, those stored messages are used. Otherwise the
    stored messages are taken up to the length of the client's history, so a
    regenerated or edited turn still drops what came after. Only user and
    assistant messages count.
    """
    earlier, newest = _split_question(messages)
    kept = await _stored_for(storage_adapter, earlier)
    return [*kept, newest] if newest is not None else kept


async def _as_stored(storage_adapter: BaseStorageAdapter, messages: list[ChatMessage]) -> bool:
    """Whether every message before the question matches the stored thread."""
    earlier, _ = _split_question(messages)
    if not earlier:
        return True
    kept = await _stored_for(storage_adapter, earlier)
    # A shorter match means the client sent more history than the thread holds.
    return len(kept) == len(earlier) and all(
        k.role == m.role and (k.content or "") == (m.content or "")
        for k, m in zip(kept, earlier, strict=True)
    )


def _split_question(
    messages: list[ChatMessage],
) -> tuple[list[ChatMessage], ChatMessage | None]:
    """Split `messages` into the history and the new question, if the last one is a question."""
    if messages and messages[-1].role == "user":
        return messages[:-1], messages[-1]
    return list(messages), None


async def _stored_for(
    storage_adapter: BaseStorageAdapter, earlier: list[ChatMessage]
) -> list[ChatMessage]:
    """The stored messages that stand for `earlier`: by id, or else by position.

    By position it may return fewer messages than `earlier` when the thread is shorter.
    """
    if not earlier:
        return []
    stored = [m for m in await storage_adapter.get_messages() if m.role in ("user", "assistant")]
    by_id = {m.id: m for m in stored if m.id}
    if all(m.id and m.id in by_id for m in earlier):
        return [by_id[m.id] for m in earlier]
    return stored[: len(earlier)]


async def _thread_storage(agent: Any, thread_id: str | None) -> BaseStorageAdapter | None:
    """Return the storage for `thread_id`, or None when the thread is not stored."""
    get_storage = getattr(agent, "get_storage_adapter", None)
    if not thread_id or get_storage is None:
        return None
    try:
        storage = await get_storage(thread_id)
        if storage is not None and await type(storage).get_thread(thread_id):
            return storage
    except Exception:
        # Without a stored thread, the history is judged as sent.
        logger.opt(exception=True).debug("No stored thread {}", thread_id)
    return None


def _working_copy(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Copy every message except the last user message.

    That message stays shared so masking it also masks the caller's copy. The
    rest of the caller's list is left alone.
    """
    question = last_user_message(messages)
    return [m if m is question else m.model_copy(deep=True) for m in messages]


async def _finish(guard: TurnGuard, result: Any) -> Any:
    """Return the result as the policy allows it, or raise Blocked."""
    if isinstance(result, BaseModel):
        text = result.model_dump_json()
    else:
        text = result if isinstance(result, str) else str(result)
    finished = await guard.finish(text)
    if finished.blocked:
        raise _blocked(guard, Stage.FINISH, finished.output)
    if not finished.replaced:
        return result
    if isinstance(result, str):
        return finished.output
    rebuilt = _rebuilt(result, finished.output)
    if rebuilt is None:
        # If the masked JSON no longer fits the schema, refuse rather than return the original.
        guard.refuse("masked result does not fit its schema")
        raise _blocked(guard, Stage.FINISH, str(Block.DEFAULT_MESSAGE))
    return rebuilt


def _blocked(guard: TurnGuard, stage: Stage, message: str) -> Blocked:
    return Blocked(message, stage=str(stage), turn_id=guard.id)


def _rebuilt(result: Any, output: str) -> BaseModel | None:
    """Rebuild a structured result from its masked JSON. None if that fails."""
    if not isinstance(result, BaseModel):
        return None
    try:
        return type(result).model_validate_json(output)
    except ValidationError:
        return None


__all__ = [
    "OpenTurn",
    "gate_conversation",
    "guard_run_override",
    "joins_turn",
    "run_turn",
    "thread_history",
    "turn",
    "validate_oversight",
]
