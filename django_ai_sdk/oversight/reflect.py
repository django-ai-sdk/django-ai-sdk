"""Reflect judges a closed turn again in the background, with the policy's reflect lenses.

A worker claims the turn under a lease, runs the lenses, and commits the answers
and the rules' effects (mask the stored answer, notify the user) in one
transaction. The task `tasks.reflect_turn` runs it.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

from asgiref.sync import sync_to_async
from django.utils import timezone

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.conf import setting
from django_ai_sdk.oversight.content import context_from_turn, load_content, unmasked_literals
from django_ai_sdk.oversight.guard.asking import (
    Answered,
    Matched,
    evaluate_rules,
    run_lenses,
    should_retry,
)
from django_ai_sdk.oversight.persist import fit_column, save_answers
from django_ai_sdk.oversight.types import (
    Answer,
    Outcome,
    Stage,
    Subject,
    TurnContext,
    join_error,
)
from django_ai_sdk.utils import as_uuid

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import datetime

    from django.db.models import Q, QuerySet

    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.models import Turn
    from django_ai_sdk.oversight.policy import ResolvedPolicy

logger = get_logger(__name__)

# How long a claim lasts before another worker may take over the turn.
REFLECT_LEASE = timedelta(minutes=10)
# Claims allowed per turn before it is marked failed.
REFLECT_ATTEMPTS = 3


def answers_from(answer_rows: list[Any]) -> dict[str, Answer]:
    """Return the latest usable answer per lens. A reviewer's answer wins over a machine's."""
    answers: dict[str, Answer] = {}
    for row in sorted(answer_rows, key=lambda row: (bool(row.reviewer), row.created_at)):
        if row.error or row.value is None:
            if row.lens not in answers:
                answers[row.lens] = Answer(error=row.error or "no value", answerer=row.answerer)
            continue
        answers[row.lens] = Answer(
            value=row.value,
            confidence=row.confidence if row.confidence is not None else 1.0,
            answerer=row.answerer,
            model=row.model,
        )
    return answers


async def areflect(turn_id: str) -> int:
    """Run the policy's reflect lenses on a closed turn. Returns the number of answers added.

    The turn is claimed first with a lease (`REFLECT_LEASE`), so a duplicate task
    finds it claimed and returns 0. If a worker dies, its lease runs out and
    `oversight_sweep` can queue the turn again.

    All results are written in one transaction, and only while this run still
    holds the lease. A failed run gives its claim back so a retry can reflect.
    After `REFLECT_ATTEMPTS` the turn is marked failed. Notifications are sent
    after the commit.
    """
    from django_ai_sdk.oversight.models import Turn

    lease = await aclaim_reflect(turn_id)
    if lease is None:
        return 0
    try:
        turn = await Turn.objects.select_related("content").filter(id=turn_id).afirst()
        if turn is None:
            return 0
        judged = await _judge(turn)
        if judged is None:
            # The policy did not load. Give the claim back so a later run can try.
            await arelease_reflect(turn_id, lease)
            return 0
        policy, view, answered, matched, found = judged
        kept = await sync_to_async(_commit_reflection)(
            turn, policy, view, answered, matched, found, lease=lease
        )
    except BaseException:
        # Nothing was written. Give the claim back so a retry can reflect.
        await arelease_reflect(turn_id, lease)
        raise
    return len(answered) if kept else 0


def claimable_q(now: datetime, pending_before: datetime | None = None) -> Q:
    """Turns reflect may claim: pending, or running with an expired lease, with attempts left.

    `pending_before` keeps only pending turns created before it, so a sweep leaves
    fresh turns to the task queued at close.
    """
    from django.db.models import Q

    from django_ai_sdk.oversight.models import Turn

    state = Turn.ReflectState
    pending = {"created_at__lt": pending_before} if pending_before is not None else {}
    return (
        Q(reflect_state=state.PENDING, **pending)
        | Q(reflect_state=state.RUNNING, reflect_lease_until__lt=now)
    ) & Q(reflect_attempts__lt=REFLECT_ATTEMPTS)


async def aclaim_reflect(turn_id: str) -> Any:
    """Claim `turn_id` for reflection. Returns the lease end time, or None if not claimed.

    A turn can be claimed when `claimable_q` matches it. The lease end time
    identifies this run's claim.
    """
    from django.db.models import F

    from django_ai_sdk.oversight.models import Turn

    now = timezone.now()
    lease = now + REFLECT_LEASE
    state = Turn.ReflectState
    claimed = await Turn.objects.filter(claimable_q(now), id=turn_id).aupdate(
        reflect_state=state.RUNNING,
        reflect_lease_until=lease,
        reflect_attempts=F("reflect_attempts") + 1,
    )
    return lease if claimed else None


async def arelease_reflect(turn_id: str, lease: Any) -> None:
    """Release a claim. The turn goes back to pending, or to failed when out of attempts."""
    from django_ai_sdk.oversight.models import Turn

    released = Turn.objects.filter(
        id=turn_id, reflect_state=Turn.ReflectState.RUNNING, reflect_lease_until=lease
    )
    try:
        await sync_to_async(mark_reflect_failed)(
            released.filter(reflect_attempts__gte=REFLECT_ATTEMPTS)
        )
        await released.aupdate(reflect_state=Turn.ReflectState.PENDING, reflect_lease_until=None)
    except Exception:
        logger.exception("Reflect could not give back its claim on turn {}", turn_id)


def _commit_reflection(
    turn: Turn,
    policy: ResolvedPolicy,
    view: TurnContext,
    answered: list[Answered],
    matched: list[Matched],
    found: dict[str, str],
    *,
    lease: Any,
) -> bool:
    """Write all of reflect's results in one transaction. False when the lease was lost.

    Notifications are queued to send after the commit.
    """
    from django.db import transaction

    from django_ai_sdk.oversight.models import Turn

    state = Turn.ReflectState
    with transaction.atomic():
        held = (
            Turn.objects.select_for_update()
            .filter(pk=turn.pk, reflect_state=state.RUNNING, reflect_lease_until=lease)
            .exists()
        )
        if not held:
            logger.warning("Reflect lost its lease on turn {}; another run reflects it", turn.id)
            return False
        save_answers(turn, policy, answered, matched, view=view, found=found)
        events = _after_effects(turn, view, answered, matched)
        Turn.objects.filter(pk=turn.pk).update(
            reflect_state=state.DONE, reflect_lease_until=None, reflected_at=timezone.now()
        )
        if events:
            from django_ai_sdk.oversight.persist import enqueue_notifications

            transaction.on_commit(lambda: enqueue_notifications(events))
    return True


def mark_reflect_failed(turns: QuerySet) -> int:
    """Mark `turns` as out of reflect attempts, noting "reflect failed" on each. Returns how many.

    Reflect gives up this way after its last failed attempt, and so does the
    `oversight_sweep` for a worker that died on it.
    """
    from django.db import transaction

    from django_ai_sdk.oversight.models import Turn

    with transaction.atomic():
        rows = list(turns.select_for_update().values_list("pk", "error"))
        for pk, error in rows:
            Turn.objects.filter(pk=pk).update(
                reflect_state=Turn.ReflectState.FAILED,
                reflect_lease_until=None,
                error=fit_column(Turn, "error", join_error(error or "", "reflect failed")),
            )
    return len(rows)


async def _judge(
    turn: Turn,
) -> tuple[ResolvedPolicy, TurnContext, list[Answered], list[Matched], dict[str, str]] | None:
    """Run reflect's lenses and rules on `turn`. None when its policy does not load."""
    from django_ai_sdk.oversight.models import LensAnswer, RuleMatch
    from django_ai_sdk.oversight.resolve import aget_policy

    policy = await aget_policy(turn.policy)
    if policy is None:
        return None
    content = await sync_to_async(load_content)(turn)
    view = context_from_turn(turn, content)
    found: dict[str, str] = {}
    if content is not None and turn.content is not None and turn.content.capture == "masked":
        # The record keeps a masked copy, but the thread keeps the text as shown.
        # Judge the thread's text so mask spans point into the stored answer.
        # What reflect finds there is masked before it is kept, and so is what
        # the turn masked at close, which a judge's reason may quote.
        masked = (view.input, view.output)
        view.input = await _stored_text(turn.user_message_id) or view.input
        view.output = await _stored_text(turn.message_id) or view.output
        found = {
            **unmasked_literals(masked[0], view.input),
            **unmasked_literals(masked[1], view.output),
        }
    existing = [row async for row in LensAnswer.objects.filter(turn=turn)]
    view.answers = answers_from(existing)
    has_text = content is not None and "input" in content and "output" in content
    if not has_text:
        logger.info("Turn {} kept no text; reflect judges only what it kept", turn.id)
    # Ask again lenses that missed the deadline inline or found their service down.
    retry = {
        row.lens
        for row in existing
        if should_retry(row.error, [(step or {}).get("error", "") for step in row.trail or []])
    }
    done = {
        rule async for rule in RuleMatch.objects.filter(turn=turn).values_list("rule", flat=True)
    }
    answered, matched = await ajudge_turn(
        policy,
        view,
        retry,
        done,
        outcome=turn.outcome,
        content=content,
        has_text=has_text,
    )
    return policy, view, answered, matched, found


async def ajudge_turn(
    policy: ResolvedPolicy,
    view: TurnContext,
    retry: Iterable[str],
    done: set[str],
    *,
    outcome: str,
    content: dict[str, Any] | None = None,
    has_text: bool = True,
    sample: bool = True,
) -> tuple[list[Answered], list[Matched]]:
    """Ask reflect's lenses on `view` and match its rules. The dry run calls this too.

    `retry` names the inline lenses to ask again, `done` the rules that already
    matched. `outcome` is the turn's outcome so far. `sample=False` asks every
    reflect lens, ignoring sample rates.
    """
    from django_ai_sdk.oversight.guard import unguarded

    # When the gate blocked the question there is no answer to judge.
    has_answer = bool(view.output.strip()) or outcome != Outcome.BLOCKED
    lenses, late = reflect_lenses(
        policy, retry, answered=has_answer, content=content, has_text=has_text
    )
    with unguarded():
        answered = await run_lenses(lenses, view, Stage.REFLECT, sample=sample)
        # No sampling, since these were due inline on every turn.
        answered += await run_lenses(late, view, Stage.REFLECT, sample=False)
    for item in answered:
        view.answers[item.lens.name] = item.answer
    # Rules treat a failed answer as unknown, not as "no".
    failed = {name for name, answer in view.answers.items() if answer.error}
    matched = evaluate_rules(policy, view.answers, Stage.REFLECT, done=done, errored=failed)
    _reflect_masks(view, answered, matched)
    return answered, matched


def reflect_lenses(
    policy: ResolvedPolicy,
    retry: Iterable[str],
    *,
    answered: bool,
    content: dict[str, Any] | None = None,
    has_text: bool = True,
) -> tuple[list[Lens], list[Lens]]:
    """Choose what reflect judges: the policy's reflect lenses, and inline lenses to ask again.

    `retry` names the inline lenses to ask again. Both lists keep only the lenses
    reflect can answer from what the turn kept (see `_judgeable`).
    """
    lenses = [
        lens
        for lens in policy.active_lenses(Stage.REFLECT)
        if _judgeable(lens, content, has_text, answered=answered)
    ]
    late = [
        policy.lenses[name]
        for name in sorted(retry)
        if name in policy.lenses
        and policy.lenses[name] not in lenses
        and _judgeable(policy.lenses[name], content, has_text, answered=answered)
    ]
    return lenses, late


def _reflect_masks(
    view: TurnContext, answered: list[Answered], matched: list[Matched]
) -> list[tuple[Matched, tuple[list[Any], str]]]:
    """Return (match, group) for each mask in the stored answer. Drop masks that change nothing.

    A mask that cannot locate what it found masks nothing here, since reflect cannot block.
    """
    from django_ai_sdk.oversight.guard.masks import collect_masks

    groups, unpointed = collect_masks(
        matched, answered, subject=Subject.OUTPUT, view=view, closed=False
    )
    for match, mask in unpointed:
        match.acted.remove(mask)
    return groups


async def _stored_text(message_id: str) -> str | None:
    """Return a thread message's stored text, or None when there is no such message."""
    messages = _messages(message_id)
    message = await messages.afirst() if messages is not None else None
    result = message.result if message is not None else None
    content = result.get("content") if isinstance(result, dict) else None
    return content if isinstance(content, str) else None


def _messages(message_id: str, *, lock: bool = False) -> QuerySet | None:
    """The thread message `message_id` as a queryset, or None when the id is empty or invalid."""
    from django_ai_sdk.conversation.models import Message

    if (pk := as_uuid(message_id)) is None:
        return None
    messages = Message.objects.filter(id=pk)
    return messages.select_for_update() if lock else messages


def _judgeable(
    lens: Lens, content: dict[str, Any] | None, has_text: bool, *, answered: bool = True
) -> bool:
    """Whether reflect can answer `lens` from what the turn kept.

    Tool call lenses never apply, since reflect has no call to show. Answer
    lenses need an answer. Without the turn's text, only a sources lens applies,
    and only when the sources were kept.
    """
    if lens.subject is Subject.TOOL_CALL:
        return False
    if not answered and lens.subject in {Subject.OUTPUT, Subject.EXCHANGE}:
        return False
    if has_text:
        return True
    return lens.subject is Subject.SOURCES and bool((content or {}).get("sources"))


def _after_effects(
    turn: Turn, view: TurnContext, answered: list[Answered], matched: list[Matched]
) -> list[tuple[dict[str, Any], list[str]]]:
    """Apply what reflect's rules can still do: mask the stored answer and notify the user.

    Runs inside reflect's transaction. Returns (event, notifier names) pairs to
    send after the commit.
    """
    from django_ai_sdk.oversight.guard import rule_notes
    from django_ai_sdk.oversight.notifiers import match_events

    groups = _reflect_masks(view, answered, matched)
    masks = [group for _, group in groups]
    rules = [match.rule.name for match, _ in groups]
    events = match_events(
        turn.id, matched, agent=turn.agent_name, channel=turn.channel, stage=str(Stage.REFLECT)
    )
    notes = rule_notes(matched, set())
    if (masks or notes) and turn.message_id:
        _update_stored(turn, view.output, masks, rules, notes)
    return events


def _update_stored(
    turn: Turn,
    judged: str,
    masks: list[tuple[list[Any], str]],
    rules: list[str],
    notes: list[dict[str, Any]],
) -> None:
    """Mask the stored answer and attach reflect's notices, which a reloaded thread shows.

    Spans index the text reflect judged, so they only apply when the stored text
    still matches it. The reasoning cannot be masked this way, so it is dropped.
    """
    from django_ai_sdk.oversight.guard.notices import notice_base
    from django_ai_sdk.oversight.patterns import mask_all

    messages = _messages(turn.message_id, lock=True)
    message = messages.first() if messages is not None else None
    if message is None or not isinstance(message.result, dict):
        return
    result = dict(message.result)
    metadata = dict(result.get("metadata") or {})
    notices = list(metadata.get("oversight") or [])
    before = len(notices)

    if masks and result.get("content") != judged:
        logger.warning("Turn {}: stored text differs from its content; not masked", turn.id)
    elif masks:
        # The user sees this notice, so include rule names only in developer detail mode.
        detail = {"rules": rules} if setting("CLIENT_DETAIL") == "full" else {}
        notices.append(
            {**notice_base(turn.id, Stage.REFLECT, Outcome.MASKED), "replaced": True, **detail}
        )
        result.update(content=mask_all(judged, masks), reasoning="")
    if notes:
        notices.append({**notice_base(turn.id, Stage.REFLECT, Outcome.FLAGGED), "notes": notes})
    if len(notices) == before:
        return
    message.result = {**result, "metadata": {**metadata, "oversight": notices}}
    message.save(update_fields=["result"])


__all__ = [
    "REFLECT_ATTEMPTS",
    "REFLECT_LEASE",
    "aclaim_reflect",
    "areflect",
    "arelease_reflect",
    "claimable_q",
    "mark_reflect_failed",
]
