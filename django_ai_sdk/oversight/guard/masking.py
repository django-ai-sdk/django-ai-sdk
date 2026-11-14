"""What a Mask rule hides in the turn's own text and in other text the model reads.

Other text (earlier messages, tool results) is judged by the gate's lenses like
the question, then masked or withheld whole; findings are recorded aside.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING

from django_ai_sdk.oversight.agents import FINGERPRINT_CHARS
from django_ai_sdk.oversight.conf import deadline_ms
from django_ai_sdk.oversight.guard.judging import (
    Answered,
    Matched,
    answer_lenses,
    cancel_all,
    errored_lenses,
    match_rules,
)
from django_ai_sdk.oversight.guard.notices import (
    BLOCKED_TEXT,
    UNCHECKED_TEXT,
    block_message,
    merge_restrict,
    removed_tool_message,
)
from django_ai_sdk.oversight.guard.stages import StageResult
from django_ai_sdk.oversight.patterns import mask_all
from django_ai_sdk.oversight.rules import Block, Mask, RestrictTools
from django_ai_sdk.oversight.types import Mode, Stage, Subject, TextSpan, TurnContext

if TYPE_CHECKING:
    from collections.abc import Iterable

    from django_ai_sdk.oversight.guard import TurnGuard
    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.policy import ResolvedPolicy
    from django_ai_sdk.oversight.rules import Action

# (spans to hide, replacement template)
MaskGroup = tuple[list[TextSpan], str]


class TurnMasks:
    """A turn's masking. The stream and the tool hook share one masking per tool result."""

    def __init__(self, guard: TurnGuard) -> None:
        self.guard = guard
        # In-flight tool result masking, keyed by call id and text hash.
        self.results: dict[str, asyncio.Future[str]] = {}
        # Masked citation source text by source key.
        self.sources: dict[str, str] = {}

    def mask_groups(self, matched: list[Matched], subject: Subject) -> list[MaskGroup]:
        """Collect the Mask groups for `subject`; a mask that cannot locate its target blocks."""
        guard = self.guard
        groups, unpointed = collect_masks(
            matched,
            guard.answered,
            subject=subject,
            context=guard.context,
            closed=guard.fails_closed,
        )
        for match, mask in unpointed:
            match.acted = [Block(match.rule.message) if a is mask else a for a in match.acted]
        return [group for _, group in groups]

    async def mask_other_text(
        self, lenses: list[Lens], text: str, stage: Stage, *, blocks: bool = False
    ) -> StageResult:
        """Judge `text` like the gate judges the question, and mask it.

        Text a mask cannot locate is replaced whole with UNCHECKED_TEXT. Only
        protective actions are kept (with `blocks`, also Block and RestrictTools),
        so other text never triggers alerts or notifications.
        """
        guard = self.guard
        judged = StageResult(stage, text=text)
        if not text:
            return judged
        context = guard.context.model_copy(update={"input": text, "answers": {}})
        judged.answered = await answer_lenses(
            lenses, context, Stage.GATE, deadline_ms=deadline_ms(stage)
        )
        answers = {o.lens.name: o.answer for o in judged.answered}
        matched = match_rules(
            guard.resolved,
            answers,
            Stage.GATE,
            done=set(),
            errored=errored_lenses(judged.answered),
            fail_closed=guard.fails_closed,
        )
        groups, unpointed = collect_masks(
            matched,
            judged.answered,
            subject=Subject.QUESTION,
            context=context,
            closed=guard.fails_closed,
        )
        for match in matched:
            acting: list[Action] = [
                a
                for a in match.acted
                if isinstance(a, Mask) or (blocks and isinstance(a, Block | RestrictTools))
            ]
            for action in acting:
                if isinstance(action, Block):
                    if not judged.blocked:
                        judged.block(block_message(match, action))
                elif isinstance(action, RestrictTools):
                    if not action.names and not judged.blocked:
                        judged.block(removed_tool_message(match))
                    judged.restrict = merge_restrict(judged.restrict, action.names)
            if acting:
                judged.matched.append(Matched(match.rule, match.stage, acting, failed=match.failed))
        if unpointed:
            judged.text = str(UNCHECKED_TEXT)
        elif groups:
            judged.text = mask_all(text, [group for _, group in groups])
        judged.changed = judged.text != text
        return judged

    async def mask_text(self, text: str) -> str:
        """Mask a tool result the way the gate masks the question. Never raises.

        A result an enforced gate rule would block is withheld whole. If it cannot
        be checked, a closed policy withholds it; an open one passes it and records
        the error.
        """
        guard = self.guard
        try:
            lenses = other_text_lenses(guard.resolved) if guard.active else []
            if not lenses or not text:
                return text
            judged = await self.mask_other_text(lenses, text, Stage.TOOL, blocks=True)
            if judged.restrict is not None:
                guard.restrict = merge_restrict(guard.restrict, judged.restrict)
            if judged.blocked:
                judged.text = str(BLOCKED_TEXT)
            if judged.text != text or any(o.errored for o in judged.answered):
                guard.record_aside(Stage.TOOL, judged.answered, judged.matched)
            return judged.text
        except Exception:
            withhold = guard.record_stage_error("tool result masking failed", block_turn=False)
            return str(UNCHECKED_TEXT) if withhold else text

    async def mask_result(self, call_id: str, text: str) -> str:
        """`mask_text` once per call id and text, so stream and tool hook get the same result."""
        if not call_id:
            return await self.mask_text(text)
        key = f"{call_id}:{hashlib.sha256(text.encode()).hexdigest()[:FINGERPRINT_CHARS]}"
        pending = self.results.get(key)
        if pending is None:
            pending = self.results[key] = asyncio.ensure_future(self.mask_text(text))
        return await asyncio.shield(pending)

    async def cancel(self) -> None:
        await cancel_all(self.results.values())


def gate_masking_lenses(policy: ResolvedPolicy) -> list[Lens]:
    names = {
        name
        for rule in policy.rules.values()
        if rule.mode is Mode.ENFORCE and any(a.kind == "mask" for a in rule.then)
        for name in rule.lenses()
    }
    return [
        lens
        for lens in policy.active_lenses(Stage.GATE, Subject.QUESTION)
        if lens.answer.kind == "entities" and lens.name in names
    ]


def other_text_lenses(policy: ResolvedPolicy) -> list[Lens]:
    """`gate_masking_lenses` plus gate lenses an enforced block or restrict_tools rule reads."""
    masking = gate_masking_lenses(policy)
    names = {
        name
        for rule in policy.rules.values()
        if rule.mode is Mode.ENFORCE
        and any(a.kind in {"block", "restrict_tools"} for a in rule.then)
        for name in rule.lenses()
    }
    return [
        lens
        for lens in policy.active_lenses(Stage.GATE, Subject.QUESTION)
        if lens in masking or lens.name in names
    ]


def collect_masks(
    matched: list[Matched],
    answered: Iterable[Answered],
    *,
    subject: Subject,
    context: TurnContext,
    closed: bool,
) -> tuple[list[tuple[Matched, MaskGroup]], list[tuple[Matched, Mask]]]:
    """Collect the (spans, template) group each Mask action in `matched` masks in `subject`.

    A mask that changes nothing is dropped, so the turn is never recorded as
    masked when nothing was. Unpointed masks are returned apart for the caller.
    """
    groups: list[tuple[Matched, MaskGroup]] = []
    unpointed: list[tuple[Matched, Mask]] = []
    text = context.text(subject)
    for match in matched:
        acted: list[Action] = []
        for action in match.acted:
            if isinstance(action, Mask):
                names = set(action.lenses) or match.rule.lenses()
                if _unpointed(answered, names, closed=closed, subject=subject, context=context):
                    unpointed.append((match, action))
                else:
                    group = (spans_for(answered, names, subject, context.input), action.template)
                    if not group[0] or mask_all(text, [group]) == text:
                        continue
                    groups.append((match, group))
            acted.append(action)
        match.acted = acted
    return groups, unpointed


def _latest(answered: Iterable[Answered], names: set[str]) -> list[Answered]:
    latest: dict[str, Answered] = {}
    for item in answered:
        if item.lens.name in names and not item.aside:
            latest[item.lens.name] = item
    return list(latest.values())


def _unpointed(
    answered: Iterable[Answered],
    names: set[str],
    *,
    closed: bool,
    subject: Subject,
    context: TurnContext,
) -> bool:
    """True when a mask cannot locate everything to hide; masking around it would leak.

    A lens failed under a closed policy, found something without spans, or has
    spans outside the text it read.
    """
    for item in _latest(answered, names):
        if item.lens.subject not in {subject, Subject.EXCHANGE}:
            continue
        if closed and item.errored:
            return True
        if item.answer.ok and item.answer.value:
            size = len(context.text(item.lens.subject))
            spans = item.answer.spans
            if not spans or not all(0 <= s.start < s.end <= size for s in spans):
                return True
    return False


def spans_for(
    answered: Iterable[Answered], names: set[str], subject: Subject, input_text: str
) -> list[TextSpan]:
    """Spans lenses `names` found in `subject`.

    An EXCHANGE lens reads input, a blank line, then output; its spans are clipped
    and shifted onto the requested part, so a span across both masks each side.
    """
    size = len(input_text)
    offset = size + 2
    spans: list[TextSpan] = []
    for item in _latest(answered, names):
        if item.lens.subject is subject:
            spans.extend(item.answer.spans)
        elif item.lens.subject is Subject.EXCHANGE and subject is Subject.ANSWER:
            spans.extend(
                s.model_copy(update={"start": max(s.start - offset, 0), "end": s.end - offset})
                for s in item.answer.spans
                if s.end > offset
            )
        elif item.lens.subject is Subject.EXCHANGE and subject is Subject.QUESTION:
            spans.extend(
                s.model_copy(update={"end": min(s.end, size)})
                for s in item.answer.spans
                if s.start < size
            )
    return spans
