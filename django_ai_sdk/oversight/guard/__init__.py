"""TurnGuard checks one turn, whatever channel it came from.

Chat, workflow steps and your own code open a guard and report what happens
(sources found, tools called, text produced). The guard is asked whether to go on
three times: at the gate, before each tool call, and at finish. Closing it writes
the turn to the record (see `recording`) and queues the background stages.

    guard = await TurnGuard.start(agent, channel="api", user=user, input=text)
    gate = await guard.gate()
    if gate.blocked:
        return gate.message
    answer = await run_my_agent(gate.text)
    finished = await guard.finish(answer)
    await guard.close()
    return finished.text

A guard for an agent without a policy, or with oversight off, is inert and
allows everything.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from asgiref.sync import sync_to_async
from django.utils import timezone

from django_ai_sdk.common import last_user_text
from django_ai_sdk.errors import Blocked
from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.agents import agent_key, declared_policy, fingerprint, turn_channel
from django_ai_sdk.oversight.conf import is_enabled, setting, validate_retention
from django_ai_sdk.oversight.content import as_source
from django_ai_sdk.oversight.guard import stages
from django_ai_sdk.oversight.guard.judging import Answered, Matched
from django_ai_sdk.oversight.guard.masking import TurnMasks
from django_ai_sdk.oversight.guard.notices import TurnNotices
from django_ai_sdk.oversight.policy import Policy, ResolvedPolicy, default_fail_mode
from django_ai_sdk.oversight.recording import record_turn
from django_ai_sdk.oversight.registry import get_resolved_policy, policy_name
from django_ai_sdk.oversight.types import (
    Capture,
    HistoryItem,
    Outcome,
    Stage,
    ToolCall,
    TurnContext,
    join_error,
    worst_outcome,
)
from django_ai_sdk.utils import scoped

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Iterable

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.oversight.guard.stages import StageResult

logger = get_logger(__name__)

_unguarded: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "django_ai_sdk_oversight_unguarded", default=False
)
_current: contextvars.ContextVar[TurnGuard | None] = contextvars.ContextVar(
    "django_ai_sdk_oversight_guard", default=None
)
_watched: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "django_ai_sdk_oversight_watched", default=None
)


def unguarded() -> contextlib.AbstractContextManager[None]:
    """Run code that must not open turns of its own, such as judges and titles."""
    return scoped(_unguarded, True)


def is_unguarded() -> bool:
    return _unguarded.get()


@contextlib.contextmanager
def watching_turns() -> Generator[list[str]]:
    """Collect the ids of the turns opened inside the block, in order.

    A run that joins an outer turn opens none, so it adds nothing.
    """
    opened: list[str] = []
    with scoped(_watched, opened):
        yield opened


def note_opened(guard: TurnGuard) -> None:
    """Tell `watching_turns()` that `guard` opened a turn of its own."""
    if (opened := _watched.get()) is not None:
        opened.append(guard.id)


def guarding(guard: TurnGuard | None) -> contextlib.AbstractContextManager[None]:
    """Make `guard` current for tool hooks and get_tools(). Tasks inherit it."""
    return scoped(_current, guard if guard is not None and guard.active else None)


@contextlib.asynccontextmanager
async def closing_turn(guard: TurnGuard, *, keep_open: bool = False) -> AsyncGenerator[None]:
    """Close `guard` when the block ends, recording an exception as the run's error.

    A disconnect is recorded as "disconnected"; `Blocked` is not an error. With
    `keep_open` the guard closes only on an exception (the Stream takes it over).
    """
    try:
        yield
    except BaseException as exc:
        if isinstance(exc, Blocked):
            await guard.close()
        elif isinstance(exc, GeneratorExit | asyncio.CancelledError):
            await guard.close(error="disconnected")
        else:
            await guard.close(error=type(exc).__name__)
        raise
    if not keep_open:
        await guard.close()


def current_guard() -> TurnGuard | None:
    """The open guard this code runs under. A task that outlived its turn gets None."""
    guard = _current.get()
    return guard if guard is not None and not guard._closed else None


def outlived_turn() -> bool:
    """True under a closed guard; its tool calls are blocked since nothing would record them."""
    guard = _current.get()
    return guard is not None and guard._closed


# How many earlier messages the lenses see with a question.
HISTORY_TURNS = 12


@dataclass
class Usage:
    """The model and token counts a turn used, summed over its model calls."""

    model: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def add(self, model: str | None, prompt: int | None, completion: int | None) -> None:
        self.model = self.model or (model or "")
        if prompt is not None:
            self.prompt_tokens = (self.prompt_tokens or 0) + prompt
        if completion is not None:
            self.completion_tokens = (self.completion_tokens or 0) + completion


@dataclass
class TurnRecordState:
    """What the record has of a turn so far. Only `recording` writes it."""

    # Written ahead after the gate, so close updates the row.
    row_written: bool = False
    answer_ids: list[str] = field(default_factory=list)
    matches_saved: int = 0


@dataclass
class TurnGuard:
    """One turn under a policy. Build it with `start()`; inert unless `active`."""

    active: bool = False
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    policy: ResolvedPolicy | None = None
    channel: str = ""
    context: TurnContext = field(default_factory=TurnContext)
    agent_fingerprint: str = ""
    thread_id: str = ""
    user_message_id: str = ""
    message_id: str = ""
    actor_id: str = ""
    answered: list[Answered] = field(default_factory=list)
    matched: list[Matched] = field(default_factory=list)
    restrict: list[str] | None = None
    # Block messages by call (name and arguments): the same call again is blocked at once.
    blocked_calls: dict[str, str] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    created_at: Any = None
    error: str = ""
    # The run's own error (crash, disconnect): the turn never finished, nothing was let through.
    run_error: str = ""
    # False for a dry run, which writes nothing.
    recording: bool = True
    # Blocked unread at the gate for exceeding MAX_INPUT_CHARS.
    too_long: bool = False
    blocked: bool = False
    # The gate ran, so a model call now counts as seen.
    gated: bool = False
    original_input: str = ""
    original_output: str = ""
    # A turn is finished once, inside its run or after it.
    finish_result: StageResult | None = None
    parent_turn_id: str = ""
    # Set by a streaming adapter to send stub results for blocked tool calls.
    on_tool_results: Callable[[list[Any]], Awaitable[None]] | None = None
    started_at: float = field(default_factory=time.perf_counter)
    _closed: bool = False
    # An errored answer carries over to later stages until the lens answers again.
    _lens_errored: dict[str, bool] = field(default_factory=dict)
    # Policy could not be chosen or loaded under FAIL_MODE "closed": every stage blocks.
    policy_unloaded: bool = False
    record: TurnRecordState = field(default_factory=TurnRecordState)
    masks: TurnMasks = field(init=False, repr=False)
    notices: TurnNotices = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.masks = TurnMasks(self)
        self.notices = TurnNotices(self)

    @property
    def resolved(self) -> ResolvedPolicy:
        """The policy; read it only when `active`."""
        assert self.policy is not None
        return self.policy

    def note_error(self, error: str) -> None:
        """Add `error` to the turn's errors. Never put content in it."""
        self.error = join_error(self.error, error)

    def block(self, error: str = "") -> None:
        """Block the turn from the caller, e.g. a masked result no longer fits its schema."""
        if not self.active:
            return
        self.blocked = True
        self.note_error(error)

    def set_history(self, messages: list[ChatMessage]) -> None:
        self.context.history = [
            HistoryItem(role=m.role, content=m.content or "")
            for m in messages[:-1]
            if m.role in {"user", "assistant"} and m.content
        ][-HISTORY_TURNS:]

    def set_sources(self, sources: Iterable[Any], *, seeded: Iterable[Any] = ()) -> None:
        """Set what retrieval found. `seeded`: sources or keys carried over from earlier turns."""
        if not self.active:
            return
        seeded_keys = {s if isinstance(s, str) else as_source(s).key for s in seeded}
        fresh = [as_source(s) for s in sources]
        for source in fresh:
            source.seeded = source.seeded or (bool(source.key) and source.key in seeded_keys)
        self.context.sources = fresh

    def allows_tool(self, name: str) -> bool:
        """`restrict` None keeps all tools, [] removes all."""
        return self.restrict is None or (bool(self.restrict) and name not in self.restrict)

    def add_tool_call(self, call: ToolCall | dict[str, Any]) -> None:
        if not self.active:
            return
        call = call if isinstance(call, ToolCall) else ToolCall.model_validate(call)
        for index, existing in enumerate(self.context.tool_calls):
            if call.id and existing.id == call.id:
                # Only what this report says. A result of 0 or False still counts.
                update = {k: v for k, v in call.model_dump().items() if v is not None and v != ""}
                update["error"] = existing.error or call.error
                self.context.tool_calls[index] = existing.model_copy(update=update)
                return
        self.context.tool_calls.append(call)

    def add_usage(self, model: str | None, prompt: int | None, completion: int | None) -> None:
        if self.active:
            self.usage.add(model, prompt, completion)

    def add_text(self, chunk: str) -> None:
        if self.active and chunk:
            self.context.output += chunk

    def keep_answers(self, answered: list[Answered]) -> None:
        """Keep answers for the record; only turn-level answers (not per tool call) feed rules."""
        for item in answered:
            self.answered.append(item)
            if item.about_turn:
                self._lens_errored[item.lens.name] = item.errored
                self.context.answers[item.lens.name] = item.answer

    @property
    def fails_closed(self) -> bool:
        return self.policy is None or self.policy.fails_closed

    def record_stage_error(self, error: str, *, block_turn: bool) -> bool:
        """Log and note `error`. True when failing closed; `block_turn` then blocks the turn."""
        logger.exception("Oversight: {} in turn {}", error, self.id)
        self.note_error(error)
        if self.fails_closed and block_turn:
            self.blocked = True
        return self.fails_closed

    def errored_lenses(self) -> set[str]:
        return {name for name, errored in self._lens_errored.items() if errored}

    def record_aside(
        self, stage: Stage, answered: list[Answered], matched: list[Matched]
    ) -> list[Matched]:
        """Record findings on other text as aside, once per lens and rule per stage.

        Spans are dropped: they point into the other text and must never mask the
        question or answer. Returns the matches added.
        """
        seen = {(o.lens.name, o.stage) for o in self.answered if o.aside}
        for item in answered:
            found = (item.answer.ok and item.answer.value) or item.errored
            if found and (item.lens.name, stage) not in seen:
                seen.add((item.lens.name, stage))
                answer = item.answer.model_copy(update={"spans": []})
                self.answered.append(Answered(item.lens, stage, answer, item.failed, aside=True))
        recorded = {(m.rule.name, m.stage) for m in self.matched if m.aside}
        added: list[Matched] = []
        for match in matched:
            if (match.rule.name, stage) not in recorded:
                recorded.add((match.rule.name, stage))
                added.append(
                    Matched(match.rule, stage, match.acted, aside=True, failed=match.failed)
                )
        self.matched.extend(added)
        return added

    @property
    def outcome(self) -> Outcome:
        blocked = [Outcome.BLOCKED] if self.too_long or self.blocked else []
        return worst_outcome(*blocked, *(m.outcome() for m in self.matched))

    @property
    def report_outcome(self) -> Outcome:
        """The outcome had every matched rule been enforced."""
        blocked = [Outcome.BLOCKED] if self.too_long or self.blocked else []
        return worst_outcome(*blocked, *(m.outcome(report=True) for m in self.matched))

    async def write_ahead(self) -> None:
        """Insert the open turn's row so a turn that never closes is left for `oversight_sweep`.

        Runs once and never raises; if it fails, close writes the whole turn.
        """
        if not self.active or self._closed or self.record.row_written or not self.recording:
            return
        try:
            await sync_to_async(record_turn)(self, closing=False)
        except Exception:
            logger.exception("Oversight could not write turn {} ahead; close writes it", self.id)

    @classmethod
    async def start(
        cls,
        agent: Any,
        *,
        channel: str | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
        thread_id: str | None = None,
        messages: list[ChatMessage] | None = None,
        input: str | None = None,
        oversight: type[Policy] | str | Literal[False] | None = None,
    ) -> TurnGuard:
        """Open a guard for one turn of `agent`. Inert unless a policy applies.

        `oversight` overrides the agent's policy (Policy subclass or name); False
        or "" opens no turn. If the policy cannot be chosen or loaded, every stage
        blocks and the turn is recorded as blocked; with FAIL_MODE "open" the turn
        runs unguarded and the failure is logged.
        """
        if input is None:
            input = last_user_text(messages or [])
        inert = cls(context=TurnContext(input=input))
        if oversight in (False, "") or is_unguarded() or not is_enabled():
            return inert

        opening = {
            "agent": agent,
            "channel": turn_channel(channel),
            "user": user,
            "thread_id": thread_id,
            "messages": messages,
            "input": input,
        }
        name = ""
        try:
            name = policy_name(oversight) if oversight else ""
            if not name:
                name = (await declared_policy(agent, channel=channel, user=user))[0]
        except Exception:
            logger.exception("Oversight could not choose a policy for {!r}", agent_key(agent))
            return cls.blocking_all(name="", error="policy could not be chosen", **opening)
        if not name:
            return inert
        try:
            resolved = await get_resolved_policy(name)
        except Exception:
            logger.exception("Oversight policy {!r} could not be loaded", name)
            resolved = None
        if resolved is None:
            return cls.blocking_all(name=name, error="policy could not be loaded", **opening)
        return cls.under_policy(resolved, **opening)

    @classmethod
    def under_policy(
        cls,
        policy: ResolvedPolicy,
        *,
        agent: Any,
        channel: str,
        user: Any,
        thread_id: str | None,
        messages: list[ChatMessage] | None,
        input: str,
    ) -> TurnGuard:
        messages = messages or []
        user_pk = getattr(user, "pk", None)
        guard = cls(
            active=True,
            policy=policy,
            channel=channel,
            thread_id=str(thread_id or ""),
            actor_id="" if user_pk is None else str(user_pk),
            created_at=timezone.now(),
            original_input=input,
        )
        guard.context = TurnContext(
            turn_id=guard.id,
            thread_id=guard.thread_id,
            channel=channel,
            agent_id=agent_key(agent),
            agent_name=str(getattr(agent, "name", "") or type(agent).__name__),
            input=input,
        )
        guard.set_history(messages)
        guard.agent_fingerprint = fingerprint(agent)
        return guard

    @classmethod
    def blocking_all(cls, *, name: str, error: str, **opening: Any) -> TurnGuard:
        """Block all under a stand-in policy so the failure is recorded. Inert if FAIL_MODE open."""
        if default_fail_mode() == "open":
            logger.error(
                "Oversight runs a turn of {!r} unguarded: its {} (FAIL_MODE is open)",
                agent_key(opening["agent"]),
                error,
            )
            return cls(context=TurnContext(input=opening["input"]))
        try:
            guard = cls.under_policy(_blocking_policy(name), **opening)
        except Exception:
            # The block cannot be recorded, so block without a record.
            logger.exception("Oversight could not open a blocking turn")
            guard = cls(active=True, policy=_blocking_policy(name), _closed=True)
        guard.policy_unloaded = True
        guard.blocked = True
        guard.note_error(error)
        logger.error(
            "Oversight blocks a turn of {!r}: its {}",
            agent_key(opening["agent"]),
            error,
        )
        return guard

    async def gate(self) -> StageResult:
        """Judge the question before the model sees it, then write the turn ahead."""
        result = await stages.judge_question(self)
        self.gated = True
        await self.write_ahead()
        return result

    async def gate_history(
        self, messages: list[ChatMessage], *, history: bool = True
    ) -> StageResult:
        """Judge history the server did not store itself. Call before `gate()`."""
        return await stages.judge_history(self, messages, history=history)

    async def tool_call(self, call: ToolCall) -> str | None:
        """Judge one tool call. Returns the block message, or None."""
        return await stages.judge_tool_call(self, call)

    async def finish(self, output: str | None = None) -> StageResult:
        """Judge the finished answer once; later calls return the same result."""
        if self.finish_result is not None:
            return self.finish_result
        result = await stages.judge_answer(self, output)
        if self.active:
            self.finish_result = result
        return result

    async def mask_text(self, text: str) -> str:
        return await self.masks.mask_text(text)

    async def mask_result(self, call_id: str, text: str) -> str:
        return await self.masks.mask_result(call_id, text)

    async def close(self, *, error: str = "") -> str | None:
        """Record the turn and queue the background stages. Never raises.

        Pending tool-result masking is cancelled first so nothing lands on a closed
        turn. `error` is the run's own failure; with it, a turn no rule acted on
        gets a blank outcome. Returns the turn id, or None (inert or failed write).
        """
        if not self.active or self._closed:
            return None
        self._closed = True
        await self.masks.cancel()
        self.note_error(error)
        self.run_error = error
        if not self.recording:
            return None
        try:
            turn_id = await sync_to_async(record_turn)(self)
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


def _blocking_policy(name: str) -> ResolvedPolicy:
    return ResolvedPolicy(
        name=name,
        capture=Capture(setting("CAPTURE")),
        retention=validate_retention(setting("RETENTION")),
        origin="unloaded",
    )
