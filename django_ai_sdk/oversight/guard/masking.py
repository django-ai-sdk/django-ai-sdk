from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING

from django_ai_sdk.oversight.agents import FINGERPRINT_CHARS
from django_ai_sdk.oversight.conf import deadline_ms, setting
from django_ai_sdk.oversight.guard.judging import (
    FAIL_CLOSED_ACTIONS,
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

MaskGroup = tuple[list[TextSpan], str]

CHUNK_OVERLAP = 200


class TurnMasks:
    def __init__(self, guard: TurnGuard) -> None:
        self.guard = guard
        self.results: dict[str, asyncio.Future[str]] = {}
        self.blocked_call_ids: set[str] = set()
        self.sources: dict[str, str] = {}

    def mask_groups(self, matched: list[Matched], subject: Subject) -> list[MaskGroup]:
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

    async def mask_other_text(self, lenses: list[Lens], text: str, stage: Stage) -> StageResult:
        guard = self.guard
        judged = StageResult(stage, text=text)
        if not text:
            return judged
        context = guard.context.model_copy(update={"input": text, "answers": {}})
        views = await judge_other_text(lenses, context, deadline_ms(stage))
        judged.answered = list({id(o): o for view in views for o in view}.values())
        enforced = {
            name
            for rule in guard.resolved.rules.values()
            if rule.mode is Mode.ENFORCE and any(a.kind in FAIL_CLOSED_ACTIONS for a in rule.then)
            for name in rule.lenses()
        }
        errored = False
        unpointed = False
        groups: list[MaskGroup] = []
        by_rule: dict[str, Matched] = {}
        for answered in views:
            failed = errored_lenses(answered)
            errored = errored or bool(failed & enforced)
            found = match_rules(
                guard.resolved,
                {o.lens.name: o.answer for o in answered},
                Stage.GATE,
                done=set(),
                errored=failed,
                fail_closed=guard.fails_closed and stage is not Stage.TOOL,
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

        def protective(actions: list[Action]) -> list[Action]:
            return [a for a in actions if a.kind in FAIL_CLOSED_ACTIONS]

        for match in by_rule.values():
            acting = protective(match.acted)
            for action in acting:
                if isinstance(action, Block):
                    if not judged.blocked:
                        judged.block(block_message(match, action))
                elif isinstance(action, RestrictTools):
                    if not action.names and not judged.blocked:
                        judged.block(removed_tool_message(match))
                    judged.restrict = merge_restrict(judged.restrict, action.names)
            if acting or (match.rule.mode is Mode.REPORT and protective(match.rule.then)):
                judged.matched.append(Matched(match.rule, match.stage, acting, failed=match.failed))
        withheld = stage is Stage.TOOL and errored and guard.fails_closed and not judged.blocked
        if unpointed or withheld:
            judged.text = str(UNCHECKED_TEXT)
        elif groups:
            judged.text = mask_all(text, groups)
        judged.changed = judged.text != text
        return judged

    async def mask_text(self, text: str) -> str:
        guard = self.guard
        try:
            lenses = other_text_lenses(guard.resolved) if guard.active else []
            if not lenses or not text:
                return text
            judged = await self.mask_other_text(lenses, text, Stage.TOOL)
            if judged.restrict is not None:
                guard.restrict = merge_restrict(guard.restrict, judged.restrict)
            if judged.blocked:
                judged.text = str(BLOCKED_TEXT)
            if judged.matched or any(o.errored for o in judged.answered):
                guard.record_other_text_findings(Stage.TOOL, judged.answered, judged.matched)
            return judged.text
        except Exception:
            withhold = guard.record_stage_error("tool result masking failed", block_turn=False)
            return str(UNCHECKED_TEXT) if withhold else text

    async def mask_result(self, call_id: str, text: str) -> str:
        if call_id in self.blocked_call_ids:
            return text
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
    answered = await answer_lenses(lenses, context, Stage.GATE, deadline_ms=deadline)
    too_long = [o.lens for o in answered if o.answer.error == TOO_LONG]
    size = chunk_chars(too_long)
    if size is None:
        return [answered]
    chunks = split_chunks(context.input, size)
    if not 1 < len(chunks) <= setting("MAX_OTHER_TEXT_CHUNKS"):
        return [answered]
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
    limits: list[int] = []
    for lens in lenses:
        for step in lens.steps():
            try:
                limit = build_judge(step).TEXT_CHARS
            except Exception:
                continue
            if limit:
                limits.append(limit)
    return min(limits, default=None)


def split_chunks(text: str, size: int) -> list[tuple[int, str]]:
    step = size - min(CHUNK_OVERLAP, size // 10)
    starts = [0]
    while starts[-1] + size < len(text):
        starts.append(starts[-1] + step)
    return [(start, text[start : start + size]) for start in starts]


def _moved(item: Answered, offset: int) -> Answered:
    spans = [
        s.model_copy(update={"start": s.start + offset, "end": s.end + offset})
        for s in item.answer.spans
    ]
    answer = item.answer.model_copy(update={"spans": spans})
    return Answered(item.lens, item.stage, answer, item.failed, item.on_other_text)


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
    masking = gate_masking_lenses(policy)
    names = {
        name
        for rule in policy.active_rules()
        if any(a.kind in FAIL_CLOSED_ACTIONS for a in rule.then)
        for name in rule.lenses()
    }
    return [
        lens
        for lens in policy.active_lenses(Stage.GATE, Subject.QUESTION)
        if lens in masking or (lens.name in names and lens.judges_tool_results)
    ]


def collect_masks(
    matched: list[Matched],
    answered: Iterable[Answered],
    *,
    subject: Subject,
    context: TurnContext,
    closed: bool,
) -> tuple[list[tuple[Matched, MaskGroup]], list[tuple[Matched, Mask]]]:
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
        if item.lens.name in names and not item.on_other_text:
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
