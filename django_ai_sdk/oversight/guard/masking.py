"""What a Mask rule hides in the turn's own text and in other text the model reads.

Other text (earlier messages, tool results) is judged by the gate's lenses like
the question, then masked or withheld whole; findings are recorded aside.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING

from django_ai_sdk.oversight.agents import FINGERPRINT_CHARS
from django_ai_sdk.oversight.conf import deadline_ms, setting
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
from django_ai_sdk.oversight.judges.base import TOO_LONG, build_judge
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

# Chunks of over-length other text overlap by this much (at most a tenth of a chunk),
# so a name on a boundary is whole in one of them.
CHUNK_OVERLAP = 200


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
        self,
        lenses: list[Lens],
        text: str,
        stage: Stage,
        *,
        blocks: bool = False,
        withhold_unchecked: bool = False,
    ) -> StageResult:
        """Judge `text` like the gate judges the question, and mask it.

        Text a mask cannot locate is replaced whole with UNCHECKED_TEXT. Only
        protective actions are kept (with `blocks`, also Block and RestrictTools),
        so other text never triggers alerts or notifications. With
        `withhold_unchecked`, a lens that fails fires no rule: under a closed
        policy the text is withheld whole instead.
        """
        guard = self.guard
        judged = StageResult(stage, text=text)
        if not text:
            return judged
        context = guard.context.model_copy(update={"input": text, "answers": {}})
        views = await judge_other_text(lenses, context, deadline_ms(stage))
        judged.answered = list({id(o): o for view in views for o in view}.values())
        errored = False
        unpointed = False
        groups: list[MaskGroup] = []
        by_rule: dict[str, Matched] = {}
        # Each chunk is decided on its own and what any of them decides holds, so a
        # finding in one chunk is never outweighed by another.
        for answered in views:
            failed = errored_lenses(answered)
            errored = errored or bool(failed)
            found = match_rules(
                guard.resolved,
                {o.lens.name: o.answer for o in answered},
                Stage.GATE,
                done=set(),
                errored=failed,
                fail_closed=guard.fails_closed and not withhold_unchecked,
            )
            located, missed = collect_masks(
                found,
                answered,
                subject=Subject.QUESTION,
                context=context,
                closed=guard.fails_closed,
            )
            groups += [group for _, group in located]
            unpointed = unpointed or bool(missed)
            for match in found:
                first = by_rule.setdefault(match.rule.name, match)
                if first is not match:
                    kinds = {a.kind for a in first.acted}
                    first.acted += [a for a in match.acted if a.kind not in kinds]
                    first.failed = first.failed and match.failed
        for match in by_rule.values():
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
        withheld = withhold_unchecked and errored and guard.fails_closed and not judged.blocked
        if unpointed or withheld:
            judged.text = str(UNCHECKED_TEXT)
        elif groups:
            judged.text = mask_all(text, groups)
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
            # What a tool or source returned and a lens could not judge (too long, down)
            # is no finding about the turn: the result is withheld instead.
            judged = await self.mask_other_text(
                lenses, text, Stage.TOOL, blocks=True, withhold_unchecked=True
            )
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


async def judge_other_text(
    lenses: list[Lens], context: TurnContext, deadline: float | None
) -> list[list[Answered]]:
    """Answers about other text, one list per part of it that rules read on its own.

    Lenses the text is too long for are asked about each chunk their judges read
    (at most MAX_OTHER_TEXT_CHUNKS), spans moved back to where they are in the
    text; each chunk's answers come with the other lenses' answers on the whole.
    A text that needs more chunks keeps its too-long answers, so FAIL_MODE decides.
    """
    answered = await answer_lenses(lenses, context, Stage.GATE, deadline_ms=deadline)
    too_long = [o.lens for o in answered if o.answer.error == TOO_LONG]
    size = chunk_chars(too_long)
    if size is None:
        return [answered]
    chunks = split_chunks(context.input, size)
    if not 1 < len(chunks) <= setting("MAX_OTHER_TEXT_CHUNKS"):
        return [answered]
    # All chunks at once, within the stage deadline: a history text judged in chunks
    # asks up to MAX_OTHER_TEXT_CHUNKS times the judges a short one does.
    parts = await asyncio.gather(
        *(
            answer_lenses(
                too_long,
                context.model_copy(update={"input": chunk}),
                Stage.GATE,
                deadline_ms=deadline,
            )
            for _, chunk in chunks
        )
    )
    whole = [o for o in answered if o.lens not in too_long]
    return [
        [*whole, *(_moved(o, offset) for o in part)]
        for (offset, _), part in zip(chunks, parts, strict=True)
    ]


def chunk_chars(lenses: Iterable[Lens]) -> int | None:
    """The fewest characters any judge of `lenses` reads, or None when none has a limit."""
    limits: list[int] = []
    for lens in lenses:
        for step in lens.steps():
            try:
                limit = build_judge(step).TEXT_CHARS
            except Exception:  # It answers its error when asked.
                continue
            if limit:
                limits.append(limit)
    return min(limits, default=None)


def split_chunks(text: str, size: int) -> list[tuple[int, str]]:
    """(offset, chunk) pairs of at most `size` characters covering `text`, overlapping."""
    step = size - min(CHUNK_OVERLAP, size // 10)
    starts = [0]
    while starts[-1] + size < len(text):
        starts.append(starts[-1] + step)
    return [(start, text[start : start + size]) for start in starts]


def _moved(item: Answered, offset: int) -> Answered:
    """`item` about a chunk, its spans moved to where they are in the whole text."""
    spans = [
        s.model_copy(update={"start": s.start + offset, "end": s.end + offset})
        for s in item.answer.spans
    ]
    answer = item.answer.model_copy(update={"spans": spans})
    return Answered(item.lens, item.stage, answer, item.failed, item.aside)


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
