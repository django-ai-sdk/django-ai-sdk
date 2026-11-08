"""Masking of the turn's own text, and of other text the model reads.

Other text, like an earlier message or a tool result, is judged by the gate's
lenses the same way as the question. It is masked or withheld whole, and the
findings are recorded aside. A tool result a gate rule refuses is withheld whole.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from django_ai_sdk.oversight.actions import Block, Mask, RestrictTools
from django_ai_sdk.oversight.agents import FINGERPRINT_CHARS
from django_ai_sdk.oversight.conf import deadline_ms
from django_ai_sdk.oversight.guard.asking import (
    Answered,
    Matched,
    cancel_all,
    errored_lenses,
    evaluate_rules,
    run_lenses,
)
from django_ai_sdk.oversight.guard.notices import (
    REFUSED_TEXT,
    UNCHECKED_TEXT,
    refusal,
    restrict_refusal,
)
from django_ai_sdk.oversight.guard.state import GuardState, merge_restrict
from django_ai_sdk.oversight.patterns import mask_all
from django_ai_sdk.oversight.types import Mode, Stage, Subject, TextSpan, TurnContext

if TYPE_CHECKING:
    from collections.abc import Iterable

    from django_ai_sdk.oversight.actions import Action
    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.policy import ResolvedPolicy

# The spans a mask hides and the template it puts in their place.
MaskGroup = tuple[list[TextSpan], str]


class Masking(GuardState):
    """Masks for the turn's own text, and for other text judged like the question."""

    def _masks(self, matched: list[Matched], subject: Subject) -> list[MaskGroup]:
        """Collect the Mask actions' groups for `subject` (see `collect_masks`).

        A mask that cannot locate what it should hide becomes a Block.
        """
        groups, unpointed = collect_masks(
            matched, self.answered, subject=subject, view=self.context, closed=self._fails_closed
        )
        for match, mask in unpointed:
            match.acted = [Block(match.rule.message) if a is mask else a for a in match.acted]
        return [group for _, group in groups]

    async def _judged(
        self, lenses: list[Lens], text: str, stage: Stage, *, blocks: bool = False
    ) -> Judged:
        """Judge `text` with `lenses` like the gate judges the question, and mask it.

        `stage` picks the deadline. Text a mask cannot locate (see `_unpointed`) is
        replaced whole with UNCHECKED_TEXT. With `blocks`, Block and RestrictTools
        actions are kept as well. Only protective actions are kept, so other text
        never triggers alerts or notifications.
        """
        judged = Judged(text=text)
        if not text:
            return judged
        view = self.context.model_copy(update={"input": text, "answers": {}})
        judged.answered = await run_lenses(lenses, view, Stage.GATE, deadline_ms=deadline_ms(stage))
        answers = {o.lens.name: o.answer for o in judged.answered}
        matched = evaluate_rules(
            self.resolved,
            answers,
            Stage.GATE,
            done=set(),
            errored=errored_lenses(judged.answered),
            fail_closed=self._fails_closed,
        )
        groups, unpointed = collect_masks(
            matched, judged.answered, subject=Subject.INPUT, view=view, closed=self._fails_closed
        )
        for match in matched:
            acting: list[Action] = [
                a
                for a in match.acted
                if isinstance(a, Mask) or (blocks and isinstance(a, Block | RestrictTools))
            ]
            for action in acting:
                if isinstance(action, Block):
                    judged.block = judged.block or refusal(match, action)
                elif isinstance(action, RestrictTools):
                    if not action.names:
                        judged.block = judged.block or restrict_refusal(match)
                    judged.restrict = merge_restrict(judged.restrict, action.names)
            if acting:
                judged.matched.append(Matched(match.rule, match.stage, acting, failed=match.failed))
        if unpointed:
            judged.text = str(UNCHECKED_TEXT)
        elif groups:
            judged.text = mask_all(text, [group for _, group in groups])
        return judged

    def _record_aside(
        self, stage: Stage, answered: list[Answered], matched: list[Matched]
    ) -> list[Matched]:
        """Record findings on other text as aside, once per lens and rule per stage.

        Spans are dropped because they point into the other text, so they never
        place masks in the question or answer. Failed lenses are recorded too.
        Returns the matches added by this call.
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

    async def mask_text(self, text: str) -> str:
        """Mask a tool result the way the gate masks the question. Never raises.

        Findings are recorded aside at the tool stage. A result an enforced gate
        rule would refuse is withheld whole, and tools it takes away stay away.
        If the result cannot be checked, a closed policy withholds it and an open
        one passes it unchanged and records the error.
        """
        try:
            lenses = gate_guarding_lenses(self.resolved) if self.active else []
            if not lenses or not text:
                return text
            judged = await self._judged(lenses, text, Stage.TOOL, blocks=True)
            if judged.restrict is not None:
                self.restrict = merge_restrict(self.restrict, judged.restrict)
            if judged.block is not None:
                judged.text = str(REFUSED_TEXT)
            if judged.text != text or any(o.errored for o in judged.answered):
                self._record_aside(Stage.TOOL, judged.answered, judged.matched)
            return judged.text
        except Exception:
            return (
                str(UNCHECKED_TEXT)
                if self._stage_failed("tool result masking failed", refuse_turn=False)
                else text
            )

    async def mask_result(self, call_id: str, text: str) -> str:
        """Mask a tool result like `mask_text`, once per call id and text.

        The stream and the tool hook both ask for the same result, and must get
        the same masked text. A different text under the same call, like a
        truncated result, is masked separately.
        """
        if not call_id:
            return await self.mask_text(text)
        key = f"{call_id}:{hashlib.sha256(text.encode()).hexdigest()[:FINGERPRINT_CHARS]}"
        pending = self._masking_results.get(key)
        if pending is None:
            pending = self._masking_results[key] = asyncio.ensure_future(self.mask_text(text))
        return await asyncio.shield(pending)

    async def _cancel_masking(self) -> None:
        await cancel_all(self._masking_results.values())


def gate_masking_lenses(policy: ResolvedPolicy) -> list[Lens]:
    """Return the gate's entity lenses that an enforced mask rule reads."""
    names = {
        name
        for rule in policy.rules.values()
        if rule.mode is Mode.ENFORCE and any(a.kind == "mask" for a in rule.then)
        for name in rule.lenses()
    }
    return [
        lens
        for lens in policy.active_lenses(Stage.GATE, Subject.INPUT)
        if lens.answer.kind == "entities" and lens.name in names
    ]


def gate_guarding_lenses(policy: ResolvedPolicy) -> list[Lens]:
    """Return `gate_masking_lenses`, plus the gate's lenses an enforced block or tool rule reads."""
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
        for lens in policy.active_lenses(Stage.GATE, Subject.INPUT)
        if lens in masking or lens.name in names
    ]


@dataclass
class Judged:
    """The result of judging other text: the text to pass on and what was found."""

    text: str
    answered: list[Answered] = field(default_factory=list)
    matched: list[Matched] = field(default_factory=list)
    # The refusal message, if an enforced rule blocks the turn.
    block: str | None = None
    # Tools removed by an enforced rule. None means none, [] means all.
    restrict: list[str] | None = None


def collect_masks(
    matched: list[Matched],
    answered: Iterable[Answered],
    *,
    subject: Subject,
    view: TurnContext,
    closed: bool,
) -> tuple[list[tuple[Matched, MaskGroup]], list[tuple[Matched, Mask]]]:
    """Collect the (spans, template) group each Mask action in `matched` masks in `subject`.

    A mask that would change nothing is dropped from its match, so the turn is
    never recorded as masked when nothing was. A mask that cannot locate what it
    should hide (see `_unpointed`) stays and is returned apart, for the caller to
    block, withhold or ignore.
    """
    groups: list[tuple[Matched, MaskGroup]] = []
    unpointed: list[tuple[Matched, Mask]] = []
    text = view.text(subject)
    for match in matched:
        acted: list[Action] = []
        for action in match.acted:
            if isinstance(action, Mask):
                names = set(action.lenses) or match.rule.lenses()
                if _unpointed(answered, names, closed=closed, subject=subject, view=view):
                    unpointed.append((match, action))
                else:
                    group = (_spans_for(answered, names, subject, view.input), action.template)
                    if not group[0] or mask_all(text, [group]) == text:
                        continue
                    groups.append((match, group))
            acted.append(action)
        match.acted = acted
    return groups, unpointed


def _latest(answered: Iterable[Answered], names: set[str]) -> list[Answered]:
    """Return the latest answer of each lens in `names`, leaving out aside answers."""
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
    view: TurnContext,
) -> bool:
    """True when a mask over lenses `names` cannot locate everything to hide in `subject`.

    That happens when a lens's latest answer failed under a closed policy, found
    something without spans, or has spans outside the text it read. Masking
    around it would leak what it found.
    """
    for item in _latest(answered, names):
        if item.lens.subject not in {subject, Subject.EXCHANGE}:
            continue
        if closed and item.errored:
            return True
        if item.answer.ok and item.answer.value:
            size = len(view.text(item.lens.subject))
            spans = item.answer.spans
            if not spans or not all(0 <= s.start < s.end <= size for s in spans):
                return True
    return False


def _spans_for(
    answered: Iterable[Answered], names: set[str], subject: Subject, input_text: str
) -> list[TextSpan]:
    """Return the spans lenses `names` found in `subject`, from each lens's latest answer.

    An EXCHANGE lens reads input, a blank line, then output. Its spans are clipped
    to the requested part and shifted onto it, so a span across both parts masks
    each side. Aside answers are ignored.
    """
    size = len(input_text)
    offset = size + 2
    spans: list[TextSpan] = []
    for item in _latest(answered, names):
        if item.lens.subject is subject:
            spans.extend(item.answer.spans)
        elif item.lens.subject is Subject.EXCHANGE and subject is Subject.OUTPUT:
            spans.extend(
                s.model_copy(update={"start": max(s.start - offset, 0), "end": s.end - offset})
                for s in item.answer.spans
                if s.end > offset
            )
        elif item.lens.subject is Subject.EXCHANGE and subject is Subject.INPUT:
            spans.extend(
                s.model_copy(update={"end": min(s.end, size)})
                for s in item.answer.spans
                if s.start < size
            )
    return spans
