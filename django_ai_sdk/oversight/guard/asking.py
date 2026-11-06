"""Ask a stage's lenses within a deadline, then find the rules that match.

Used by every guard stage, by reflect and by `testing.dry_run`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.actions import Block
from django_ai_sdk.oversight.judges.base import (
    ask,
    is_unavailable,
    sampled,
)
from django_ai_sdk.oversight.judges.classifier import batched
from django_ai_sdk.oversight.types import (
    INLINE_STAGES,
    Answer,
    Mode,
    Outcome,
    Stage,
    Subject,
    rank,
    worst_outcome,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from collections.abc import Set as AbstractSet

    from django_ai_sdk.oversight.actions import Action
    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.policy import ResolvedPolicy
    from django_ai_sdk.oversight.rules import Rule
    from django_ai_sdk.oversight.types import TurnContext

logger = get_logger(__name__)

STAGE_ORDER = tuple(Stage)


@dataclass
class Answered:
    lens: Lens
    stage: Stage
    answer: Answer
    # The chain hit an error or the deadline after only unsure answers. The lens
    # counts as failed for FAIL_MODE, but the unsure answer is kept for the record.
    failed: bool = False
    # About other text, like an earlier message or a tool result. Recorded, but
    # never used to place masks in the question or answer.
    aside: bool = False

    @property
    def errored(self) -> bool:
        """True when the lens failed at this stage."""
        return self.failed or bool(self.answer.error)


# Actions a failed lens may trigger under FAIL_MODE "closed". Only protective ones.
FAIL_CLOSED_ACTIONS = frozenset({"block", "mask", "restrict_tools"})


@dataclass
class Matched:
    """A rule that matched at a stage, and which of its actions were carried out."""

    rule: Rule
    stage: Stage
    # Actions this stage can carry out. Empty unless the rule is enforced.
    acted: list[Action] = field(default_factory=list)
    # Matched on other text, like an earlier message or a tool result. It counts
    # toward the outcome but does not stop the rule from judging the turn itself.
    aside: bool = False
    # Matched only because a lens failed under FAIL_MODE "closed". The person sees
    # the generic refusal, not the rule's message or notice.
    failed: bool = False

    def outcome(self, *, report: bool = False) -> Outcome:
        """The outcome this match gives the turn. With `report`, as if the rule were enforced."""
        actions = self.rule.then if report else self.acted
        return worst_outcome(*(a.outcome for a in actions))


async def answer_lens(
    lens: Lens,
    view: TurnContext,
    stage: Stage,
    *,
    sample: bool = True,
    progress: list[Answer] | None = None,
) -> Answer | None:
    """Ask a lens's steps in order (its patterns, check or judges) until one is sure.

    Only reflect samples. Inline stages always ask. `sample=False` ignores the
    sample rate, for `testing.dry_run`. `progress` collects each answer as it
    arrives, so a caller that cancels still has them.
    """
    if not lens.applies(view):
        return None
    if (
        sample
        and stage not in INLINE_STAGES
        and not sampled(f"{view.turn_id}:{lens.name}", lens.sample)
    ):
        return None
    steps = lens.steps()
    if not steps:
        return None
    return await _first_sure(steps, lens, view, progress if progress is not None else [])


async def _first_sure(
    steps: list[Any], lens: Lens, view: TurnContext, asked: list[Answer]
) -> Answer | None:
    """Return the first answer above `min_confidence`, else the last usable one.

    Confidence scales differ per judge, so they are never compared. Lenses list
    judges from quick to thorough, so the last one is trusted most. Earlier
    answers go in the result's `trail`, in the order asked.
    """
    for step in steps:
        result = await ask(step, lens, view)
        if result is None:
            continue
        if lens.settles(result):
            result.trail = list(asked)
            asked.append(result)
            return result
        asked.append(result)
    if not any(a.ok for a in asked):
        return asked[-1] if asked else None
    return _last_usable(asked)


def _ended_unsure(lens: Lens, answer: Answer | None, asked: list[Answer]) -> bool:
    """True when the chain ended on an error and `answer`, the last usable one, is unsure."""
    if answer is None or not answer.ok or lens.settles(answer):
        return False
    last = asked[-1] if asked else None
    return last is not None and bool(last.error)


def should_retry(error: str, trail_errors: Sequence[str] = ()) -> bool:
    """True when reflect should ask a lens again because of the deadline or an outage.

    `trail_errors` are the errors on the answer's trail; only the last one counts.
    A usable answer kept at the deadline has "deadline" there. One kept while a
    later judge was down has that judge's error.
    """
    last_error = trail_errors[-1] if trail_errors else ""
    return any(e == "deadline" or is_unavailable(e) for e in (error, last_error))


async def run_lenses(
    lenses: list[Lens],
    view: TurnContext,
    stage: Stage,
    *,
    deadline_ms: float | None = None,
    sample: bool = True,
) -> list[Answered]:
    """Answer `lenses` concurrently. Lenses still running at the deadline answer "deadline".

    A lens cut off after an unsure answer keeps it, with "deadline" last in its
    trail, so reflect still asks the slower judges. It counts as failed.
    Running tasks are cancelled and awaited, so no judge outlives its stage.
    """
    if not lenses:
        return []
    tasks: dict[asyncio.Future, tuple[Lens, list[Answer]]] = {}
    # Tasks copy the batch context when created, so it can close before they run.
    with batched(deadline_ms):
        for lens in lenses:
            progress: list[Answer] = []
            task = asyncio.ensure_future(
                answer_lens(lens, view, stage, sample=sample, progress=progress)
            )
            tasks[task] = (lens, progress)
    try:
        done, pending = await asyncio.wait(
            tasks, timeout=deadline_ms / 1000 if deadline_ms else None
        )
    except asyncio.CancelledError:
        await cancel_all(tasks)
        raise
    await cancel_all(pending)
    answered: list[Answered] = []
    for task in pending:
        lens, progress = tasks[task]
        answer = _last_usable(progress, Answer(error="deadline"))
        # Check against the trail, which now ends with "deadline".
        failed = _ended_unsure(lens, answer, answer.trail)
        answered.append(Answered(lens, stage, answer, failed=failed))
    for task in done:
        lens, progress = tasks[task]
        try:
            answer = task.result()
        except Exception as exc:
            logger.opt(exception=exc).warning("Lens {} failed", lens.name)
            answer = Answer(error=type(exc).__name__)
        if answer is not None:
            failed = _ended_unsure(lens, answer, progress)
            answered.append(Answered(lens, stage, answer, failed=failed))
    order = {lens.name: index for index, lens in enumerate(lenses)}
    return sorted(answered, key=lambda o: order[o.lens.name])


async def cancel_all(tasks: Iterable[asyncio.Future]) -> None:
    """Cancel `tasks` and wait until they have stopped."""
    running = [task for task in tasks if not task.done()]
    for task in running:
        task.cancel()
    if running:
        await asyncio.gather(*running, return_exceptions=True)


def _last_usable(asked: list[Answer], *extra: Answer) -> Answer:
    """The last usable answer, with the others and then `extra` as its trail.

    Without a usable one, the first of `extra` (at the deadline, the "deadline" error).
    """
    kept = next((a for a in reversed(asked) if a.ok), None)
    if kept is None:
        return extra[0]
    return kept.model_copy(update={"trail": [*(a for a in asked if a is not kept), *extra]})


def _still_to_answer(policy: ResolvedPolicy, name: str, stage: Stage, answered: set[str]) -> bool:
    """True when lens `name` may still answer at a later stage of this turn."""
    lens = policy.lenses.get(name)
    if lens is None or not lens.enabled or name in answered:
        return False
    here = rank(STAGE_ORDER, stage)
    return any(rank(STAGE_ORDER, s) > here for s in lens.stages)


def evaluate_rules(
    policy: ResolvedPolicy,
    answers: dict[str, Answer],
    stage: Stage,
    *,
    done: set[str],
    errored: AbstractSet[str] = frozenset(),
    fail_closed: bool = False,
) -> list[Matched]:
    """Return rules that match now and have not matched earlier in this turn.

    A rule waits until every lens it reads has had its chance, so a gate lens and
    a reflect lens can share a rule.

    `errored` names lenses that failed this turn. A test on one of them is
    unknown, never true (`Condition.evaluate`), unless it asks whether the lens
    answered. With `fail_closed`, a failed lens triggers an enforced rule that
    reads it, but only when that rule would block, mask or restrict tools here.
    Rules that only alert, queue or notify never fire because of a failure.
    """
    answered = set(answers) | set(errored)
    matched: list[Matched] = []
    for rule in policy.active_rules():
        if rule.name in done:
            continue
        reads = rule.lenses()
        if any(_still_to_answer(policy, name, stage, answered) for name in reads):
            continue
        acted = (
            [a for a in rule.ordered_actions() if stage in a.effective]
            if rule.mode is Mode.ENFORCE
            else []
        )
        if rule.mode is Mode.ENFORCE and stage is Stage.TOOL and _reads_tool_call(policy, reads):
            # A tool call cannot be partly masked, so a mask rule refuses the call.
            if any(a.kind == "mask" for a in rule.then) and not any(
                a.kind == "block" for a in acted
            ):
                acted.append(Block(rule.message))
        failed = (
            fail_closed
            and bool(reads & set(errored))
            and any(a.kind in FAIL_CLOSED_ACTIONS for a in acted)
        )
        if rule.when.evaluate(answers, errored):
            matched.append(Matched(rule, stage, acted))
        elif failed:
            matched.append(Matched(rule, stage, acted, failed=True))
    return matched


def errored_lenses(answered: list[Answered]) -> set[str]:
    """Return the names of lenses whose answer failed."""
    return {o.lens.name for o in answered if o.errored}


def _reads_tool_call(policy: ResolvedPolicy, names: AbstractSet[str]) -> bool:
    return any(
        (lens := policy.lenses.get(name)) is not None and lens.subject is Subject.TOOL_CALL
        for name in names
    )


__all__ = [
    "Matched",
    "Answered",
    "answer_lens",
    "errored_lenses",
    "evaluate_rules",
    "run_lenses",
    "should_retry",
]
