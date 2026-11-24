"""How each inline stage (gate, history, tool call, finish) judges its text and acts on the turn."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from django_ai_sdk.oversight.conf import deadline_ms, setting
from django_ai_sdk.oversight.guard.judging import (
    answer_lenses,
    errored_lenses,
    judge_stage,
    match_rules,
)
from django_ai_sdk.oversight.guard.notices import (
    TOO_LONG_MESSAGE,
    TOOL_UNAVAILABLE_MESSAGE,
    UNCHECKED_TEXT,
    first_block_message,
    merge_restrict,
    notice_base,
    removed_tool_message,
    restricted_tools,
)
from django_ai_sdk.oversight.patterns import mask_all
from django_ai_sdk.oversight.rules import Block
from django_ai_sdk.oversight.types import Outcome, Stage, Subject

if TYPE_CHECKING:
    from collections.abc import Callable

    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.oversight.guard import TurnGuard
    from django_ai_sdk.oversight.guard.judging import Answered, Matched
    from django_ai_sdk.oversight.types import Answer, ToolCall

# Caps one turn's cost: older items are withheld unread, or passed when failing open.
MAX_HISTORY_JUDGED = 20
HISTORY_CONCURRENCY = 4


@dataclass
class StageResult:
    """What one stage decided about its text."""

    stage: Stage = Stage.GATE
    # The text as it may go on: masked, withheld or replaced by the block message.
    text: str = ""
    changed: bool = False
    blocked: bool = False
    message: str = ""
    # The `data-oversight` payload, if there is anything to say.
    notice: dict[str, Any] | None = None
    # None means no tools removed, [] means all.
    restrict: list[str] | None = None
    answered: list[Answered] = field(default_factory=list)
    matched: list[Matched] = field(default_factory=list)
    # The tool call's name, for a tool stage of a dry run.
    tool: str = ""

    @property
    def answers(self) -> dict[str, Answer]:
        return {o.lens.name: o.answer for o in self.answered}

    @property
    def rules(self) -> list[str]:
        """Rules that matched at this stage, whether they acted or not."""
        return [m.rule.name for m in self.matched]

    def block(self, message: str) -> StageResult:
        self.blocked = True
        self.message = message
        return self


async def judge_question(guard: TurnGuard) -> StageResult:
    result = StageResult(Stage.GATE, text=guard.context.input)
    if not guard.active:
        return result
    if guard.policy_unloaded:
        return _block_question(guard, result, str(Block.DEFAULT_MESSAGE))
    limit = setting("MAX_INPUT_CHARS")
    if limit and len(guard.context.input) > limit:
        # Block now, because later stages like reflect would only read part of it.
        guard.too_long = True
        return _block_question(guard, result, str(TOO_LONG_MESSAGE))
    try:
        matched = await judge_stage(
            guard, Stage.GATE, subjects={Subject.QUESTION}, deadline=deadline_ms(Stage.GATE)
        )
        guard.matched.extend(matched)
        masks = guard.masks.mask_groups(matched, Subject.QUESTION)
        block = first_block_message(matched)
        # Tools taken away by judge_history stay taken away.
        guard.restrict = restricted_tools(matched, guard.restrict)
        if block is not None:
            result.block(block)
        if masks:
            masked = mask_all(guard.context.input, masks)
            result.changed = masked != guard.context.input
            guard.context.input = masked
        result.text = guard.context.input
        result.notice = guard.notices.stage_notice(
            Stage.GATE, matched, result.blocked, result.message
        )
        return result
    except Exception:
        # A broken lens or rule follows the fail mode, as in the other stages.
        if guard.record_stage_error("gate failed", block_turn=True):
            return _block_question(guard, result, str(Block.DEFAULT_MESSAGE))
        return result


def _block_question(guard: TurnGuard, result: StageResult, message: str) -> StageResult:
    result.block(message)
    result.notice = {**notice_base(guard.id, Stage.GATE, Outcome.BLOCKED), "message": message}
    return result


async def judge_history(
    guard: TurnGuard,
    earlier: list[ChatMessage],
    question: ChatMessage | None = None,
    *,
    history: bool = True,
) -> StageResult:
    """Judge earlier messages and attachment context that come with the question.

    Each item is masked in place or withheld whole; `history=False` judges only attachments.
    An enforced block or remove-all-tools match blocks: history is resent every turn, so
    dropping tools would silently strip every later turn (alert, queue, notify are skipped
    too). Findings are recorded aside and never steer the question's masks or rules.
    """
    result = StageResult(Stage.GATE, text=guard.context.input)
    if not guard.active:
        return result
    # Set again below once judging has masked it.
    guard.set_history(earlier)
    if guard.policy_unloaded:
        return _block_question(guard, result, str(Block.DEFAULT_MESSAGE))
    lenses = guard.resolved.active_lenses(Stage.GATE, Subject.QUESTION)
    items = _history_items(earlier, question, history=history) if lenses else []
    if not items:
        return result
    unread, items = items[:-MAX_HISTORY_JUDGED], items[-MAX_HISTORY_JUDGED:]
    limit = setting("MAX_INPUT_CHARS")
    if limit and len(guard.context.input) + sum(len(get()) for get, _ in items) > limit:
        guard.too_long = True
        return _block_question(guard, result, str(TOO_LONG_MESSAGE))
    if unread and guard.fails_closed:
        for _get, put in unread:
            put(str(UNCHECKED_TEXT))
    elif unread:
        guard.note_error("history passed unchecked")
    slots = asyncio.Semaphore(HISTORY_CONCURRENCY)

    async def judge(text: str) -> StageResult:
        async with slots:
            return await guard.masks.mask_other_text(lenses, text, Stage.GATE)

    try:
        judged = await asyncio.gather(*(judge(get()) for get, _ in items))
    except Exception:
        if guard.record_stage_error("history check failed", block_turn=True):
            return _block_question(guard, result, str(Block.DEFAULT_MESSAGE))
        return result
    found: list[Matched] = []
    for (_read, put), one in zip(items, judged, strict=True):
        put(one.text)
        found.extend(one.matched)
        if one.blocked and not result.blocked:
            result.block(one.message)
        if one.restrict is not None:
            guard.restrict = merge_restrict(guard.restrict, one.restrict)
    recorded = guard.record_aside(Stage.GATE, [o for one in judged for o in one.answered], found)
    guard.set_history(earlier)
    if result.blocked:
        result.notice = guard.notices.stage_notice(Stage.GATE, recorded, True, result.message)
        # The gate will not run after a block, so write the turn ahead here.
        await guard.write_ahead()
    return result


def _history_items(
    earlier: list[ChatMessage], question: ChatMessage | None, *, history: bool
) -> list[tuple[Callable[[], str], Callable[[str], None]]]:
    """(read, write back) pairs for each text to judge.

    Attachment context is rebuilt from the document each turn and never stored,
    so a vouched history does not cover it.
    """
    items: list[tuple[Callable[[], str], Callable[[str], None]]] = []
    for message in [*earlier, question] if question else earlier:
        said = message.role in {"user", "assistant"} and message.content
        if history and said and message is not question:
            items.append(
                (
                    lambda m=message: m.content,
                    lambda text, m=message: setattr(m, "content", text),
                )
            )
        for attachment in message.attachments:
            if attachment.context:
                items.append(
                    (
                        lambda a=attachment: a.context,
                        lambda text, a=attachment: setattr(a, "context", text),
                    )
                )
    return items


async def judge_tool_call(guard: TurnGuard, call: ToolCall) -> str | None:
    if not guard.active:
        return None
    if guard.policy_unloaded:
        return str(Block.DEFAULT_MESSAGE)
    try:
        key = f"{call.name}:{json.dumps(call.arguments, sort_keys=True, default=str)}"
        if key in guard.blocked_calls:
            return guard.blocked_calls[key]
        if not guard.allows_tool(call.name):
            return str(TOOL_UNAVAILABLE_MESSAGE)
        lenses = guard.resolved.active_lenses(Stage.TOOL, Subject.TOOL_CALL)
        if not lenses:
            return None
        context = guard.context.model_copy(update={"tool_call": call})
        answered = await answer_lenses(
            lenses, context, Stage.TOOL, deadline_ms=deadline_ms(Stage.TOOL)
        )
        guard.keep_answers(answered)
        # Keep this call's answers out of context.answers so the next call is judged on its own.
        answers = {o.lens.name: o.answer for o in answered}
        tool_lenses = {lens.name for lens in lenses}
        # Failures of other lenses still count. A tool lens failure counts only for this call.
        errored = {name for name in guard.errored_lenses() if name not in tool_lenses}
        # Judge every call, not just the first, so a rule blocks a changed call too.
        matched = [
            m
            for m in match_rules(
                guard.resolved,
                {**guard.context.answers, **answers},
                Stage.TOOL,
                done=set(),
                errored=errored | errored_lenses(answered),
                fail_closed=guard.fails_closed,
            )
            if m.rule.lenses() & tool_lenses
        ]
        # One recorded match per rule, however many calls it blocked.
        recorded = {m.rule.name for m in guard.matched if not m.aside}
        guard.matched.extend(m for m in matched if m.rule.name not in recorded)
        block = first_block_message(matched)
        guard.restrict = restricted_tools(matched, guard.restrict)
        if block is not None:
            guard.blocked_calls[key] = block
            return block
        if not guard.allows_tool(call.name):
            restricting = next(
                (m for m in matched if any(a.kind == "restrict_tools" for a in m.acted)), None
            )
            return (
                removed_tool_message(restricting) if restricting else str(TOOL_UNAVAILABLE_MESSAGE)
            )
        return None
    except Exception:
        fails_closed = guard.record_stage_error("tool check failed", block_turn=False)
        return str(Block.DEFAULT_MESSAGE) if fails_closed else None


async def judge_answer(guard: TurnGuard, output: str | None) -> StageResult:
    if output is not None:
        guard.context.output = output
    guard.original_output = guard.original_output or guard.context.output
    result = StageResult(Stage.FINISH, text=guard.context.output)
    if not guard.active:
        return result
    if guard.policy_unloaded:
        return _withhold_answer(guard, result)
    try:
        matched = await judge_stage(
            guard,
            Stage.FINISH,
            subjects={Subject.ANSWER, Subject.EXCHANGE, Subject.SOURCES},
            deadline=deadline_ms(Stage.FINISH),
        )
        guard.matched.extend(matched)
        masks = guard.masks.mask_groups(matched, Subject.ANSWER)
        block = first_block_message(matched)
        if block is not None:
            result.block(block)
            result.changed = True
            result.text = block
        elif masks:
            masked = mask_all(result.text, masks)
            result.changed = masked != result.text
            result.text = masked
        if result.changed:
            guard.context.output = result.text
        result.notice = guard.notices.stage_notice(
            Stage.FINISH, matched, result.blocked, result.message, result
        )
        return result
    except Exception:
        # Keep the answer, unless the policy fails closed.
        if guard.record_stage_error("finish failed", block_turn=True):
            return _withhold_answer(guard, result)
        return result


def _withhold_answer(guard: TurnGuard, result: StageResult) -> StageResult:
    result.block(str(Block.DEFAULT_MESSAGE))
    result.changed = True
    result.text = guard.context.output = result.message
    return result
