"""The state a guard builds up during a turn.

GuardState holds TurnGuard's fields and the helpers every stage shares: noting
errors, asking a stage's lenses (`_stage`) and writing the turn ahead of close.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from asgiref.sync import sync_to_async

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.content import as_source
from django_ai_sdk.oversight.guard.asking import Answered, Matched, evaluate_rules, run_lenses
from django_ai_sdk.oversight.persist import persist_turn
from django_ai_sdk.oversight.types import (
    HistoryItem,
    Outcome,
    Stage,
    Subject,
    ToolCall,
    TurnContext,
    join_error,
    worst_outcome,
)

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Awaitable, Callable, Iterable

    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.oversight.guard import TurnGuard
    from django_ai_sdk.oversight.policy import ResolvedPolicy

logger = get_logger(__name__)

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
class GuardState:
    """What a TurnGuard knows about its turn so far."""

    active: bool = False
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    policy: ResolvedPolicy | None = None
    channel: str = ""
    context: TurnContext = field(default_factory=TurnContext)
    agent_fingerprint: str = ""
    thread_id: str = ""
    user_message_id: str = ""
    message_id: str = ""
    # The asking user's id, "" when nobody was signed in.
    actor_id: str = ""
    answered: list[Answered] = field(default_factory=list)
    matched: list[Matched] = field(default_factory=list)
    restrict: list[str] | None = None
    usage: Usage = field(default_factory=Usage)
    created_at: Any = None
    # Errors in the turn, joined with "; ".
    error: str = ""
    # The run's own error passed to close(), like a crash or disconnect. The turn
    # never finished, so nothing was let through.
    run_error: str = ""
    # False for a dry run, which writes nothing.
    recording: bool = True
    # The gate ran, so a model call now counts as seen (`oversight.calls`).
    gated: bool = False
    # Refused unread at the gate for exceeding MAX_INPUT_CHARS.
    too_long: bool = False
    # Set when a stage failed closed, the policy did not load, or the caller called
    # `refuse`. The turn is recorded as blocked.
    refused: bool = False
    original_input: str = ""
    original_output: str = ""
    # data-oversight payloads waiting to be streamed.
    notices: list[dict[str, Any]] = field(default_factory=list)
    # The enclosing turn when this one has its own policy, like a subagent or nested run().
    parent_turn_id: str = ""
    # Set by a streaming adapter to send stub results for refused tool calls, so the
    # chat shows them as refused instead of still running.
    on_tool_results: Callable[[list[Any]], Awaitable[None]] | None = None
    _started: float = field(default_factory=time.perf_counter)
    _closed: bool = False
    # In-flight tool result masking, keyed by call id and text hash. Shared by the
    # stream and the tool hook.
    _masking_results: dict[str, asyncio.Future[str]] = field(default_factory=dict)
    # Whether each lens's latest answer errored. An error carries over to later
    # stages until the lens answers again.
    _lens_errored: dict[str, bool] = field(default_factory=dict)
    # Rules whose notice was already sent this turn.
    _told: set[str] = field(default_factory=set)
    # The policy could not be chosen or loaded under FAIL_MODE "closed". Every stage refuses.
    _unloaded: bool = False
    # The row was written ahead after the gate, so close updates it.
    _written: bool = False
    # Ids of answers already written, in order, and how many matches were saved.
    _answer_ids: list[str] = field(default_factory=list)
    _matched_saved: int = 0

    @property
    def resolved(self) -> ResolvedPolicy:
        """The policy of an active guard, which always has one. Read it only when `active`."""
        assert self.policy is not None
        return self.policy

    def note_error(self, error: str) -> None:
        """Add `error` to the turn's errors. Never put content in it."""
        self.error = join_error(self.error, error)

    def refuse(self, error: str = "") -> None:
        """Mark the turn as refused by the caller, recorded as blocked with `error`.

        For results oversight allowed that the caller still cannot use, like a
        masked result that no longer fits its schema.
        """
        if not self.active:
            return
        self.refused = True
        self.note_error(error)

    def set_history(self, messages: list[ChatMessage]) -> None:
        """Set the earlier conversation the lenses see, keeping the latest HISTORY_TURNS."""
        self.context.history = [
            HistoryItem(role=m.role, content=m.content or "")
            for m in messages[:-1]
            if m.role in {"user", "assistant"} and m.content
        ][-HISTORY_TURNS:]

    def set_sources(self, sources: Iterable[Any], *, seeded: Iterable[Any] = ()) -> None:
        """Set what retrieval found, as NumberedSource, Source or dicts.

        `seeded` lists sources carried over from earlier turns, as sources or keys
        (a SourceRegistry keeps keys).
        """
        if not self.active:
            return
        seeded_keys = {s if isinstance(s, str) else as_source(s).key for s in seeded}
        fresh = [as_source(s) for s in sources]
        for source in fresh:
            source.seeded = source.seeded or (bool(source.key) and source.key in seeded_keys)
        self.context.sources = fresh

    def allows_tool(self, name: str) -> bool:
        """Whether tool `name` is still available. `restrict` None keeps all, [] removes all."""
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
        """Add streamed answer text; finish judges the whole answer."""
        if self.active and chunk:
            self.context.output += chunk

    # Asking.

    async def _stage(
        self,
        stage: Stage,
        *,
        subjects: set[Subject],
        deadline: float | None,
    ) -> list[Matched]:
        """Ask the stage's lenses on `subjects` and return the rules that match now."""
        lenses = [lens for lens in self.resolved.active_lenses(stage) if lens.subject in subjects]
        answered = await run_lenses(lenses, self.context, stage, deadline_ms=deadline)
        self._keep_answers(answered)
        return evaluate_rules(
            self.resolved,
            self.context.answers,
            stage,
            done={m.rule.name for m in self.matched if not m.aside},
            errored=self._errored_lenses(),
            fail_closed=self._fails_closed,
        )

    def _keep_answers(self, answered: list[Answered]) -> None:
        """Store the stage's answers for the record and for rule evaluation.

        Tool call lenses judge one call each. The record keeps every call's answer,
        but rules never read them as the turn's answer or failure. Other lenses keep
        their latest answer and failure state.
        """
        for item in answered:
            self.answered.append(item)
            if item.lens.subject is not Subject.TOOL_CALL:
                self._lens_errored[item.lens.name] = item.errored
                self.context.answers[item.lens.name] = item.answer

    @property
    def _fails_closed(self) -> bool:
        """True when a failed lens should trigger the policy's protective rules."""
        return self.policy is None or self.policy.fails_closed

    def _stage_failed(self, error: str, *, refuse_turn: bool) -> bool:
        """Log the exception being handled and note `error` on the turn.

        Returns True when the policy fails closed, so the caller refuses. With
        `refuse_turn` the whole turn is then refused. Without it (a tool call or a
        tool result) only that one call is.
        """
        logger.exception("Oversight: {} in turn {}", error, self.id)
        self.note_error(error)
        if self._fails_closed and refuse_turn:
            self.refused = True
        return self._fails_closed

    def _errored_lenses(self) -> set[str]:
        """Return lenses whose latest answer this turn errored."""
        return {name for name, errored in self._lens_errored.items() if errored}

    # Recording.

    @property
    def outcome(self) -> Outcome:
        return self._outcome(report=False)

    @property
    def report_outcome(self) -> Outcome:
        """The outcome had every matched rule been enforced."""
        return self._outcome(report=True)

    def _outcome(self, *, report: bool) -> Outcome:
        refused = [Outcome.BLOCKED] if self.too_long or self.refused else []
        return worst_outcome(*refused, *(m.outcome(report=report) for m in self.matched))

    async def write_ahead(self) -> None:
        """Insert the turn's row now, still open, with the answers and matches so far.

        The gate calls this so a turn that never closes (a killed worker, an unread
        response) still leaves a row for `oversight_sweep` to mark interrupted.
        Runs once and never raises. If it fails, close writes the whole turn.
        """
        if not self.active or self._closed or self._written or not self.recording:
            return
        # GuardState is only ever used as part of TurnGuard.
        guard = cast("TurnGuard", self)
        try:
            await sync_to_async(persist_turn)(guard, closing=False)
        except Exception:
            logger.exception("Oversight could not write turn {} ahead; close writes it", self.id)


def merge_restrict(restrict: list[str] | None, names: list[str]) -> list[str]:
    """Add `names` to the removed tools `restrict`. None means none removed, [] means all."""
    if not names or restrict == []:
        return []
    return sorted({*(restrict or []), *names})
