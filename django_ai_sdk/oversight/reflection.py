from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

from asgiref.sync import sync_to_async
from django.db import transaction
from django.db.models import F, Q
from django.utils import timezone

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.conf import setting
from django_ai_sdk.oversight.content import context_from_turn, load_content, unmasked_literals
from django_ai_sdk.oversight.guard import unguarded
from django_ai_sdk.oversight.guard.judging import (
    Answered,
    Matched,
    answer_lenses,
    match_rules,
    should_retry,
)
from django_ai_sdk.oversight.guard.masking import collect_masks
from django_ai_sdk.oversight.guard.notices import notice_base, rule_notes
from django_ai_sdk.oversight.notifiers import notify_events
from django_ai_sdk.oversight.patterns import mask_all
from django_ai_sdk.oversight.recording import fit_column, record_answers
from django_ai_sdk.oversight.registry import get_resolved_policy
from django_ai_sdk.oversight.types import (
    Answer,
    Outcome,
    Stage,
    Subject,
    TextSpan,
    TurnContext,
    join_error,
)
from django_ai_sdk.utils import as_uuid

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from datetime import datetime

    from django.db.models import QuerySet

    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.models import Turn
    from django_ai_sdk.oversight.policy import ResolvedPolicy

logger = get_logger(__name__)

REFLECT_LEASE = timedelta(minutes=10)
REFLECT_ATTEMPTS = 3

MatchedMaskGroup = tuple[Matched, tuple[list[Any], str]]


def answers_from(answer_rows: list[Any], lenses: Mapping[str, Lens]) -> dict[str, Answer]:
    answers: dict[str, Answer] = {}
    for row in sorted(answer_rows, key=lambda row: row.created_at):
        lens = lenses.get(row.lens)
        if lens is not None and row.stage not in lens.stages:
            continue
        if row.error or row.value is None:
            if row.lens not in answers:
                answers[row.lens] = Answer(error=row.error or "no value", answerer=row.answerer)
            continue
        answers[row.lens] = Answer(
            value=row.value,
            confidence=row.confidence if row.confidence is not None else 1.0,
            answerer=row.answerer,
            model=row.model,
            trail=[
                Answer(
                    value=step.get("value"),
                    model=step.get("model") or "",
                    error=step.get("error") or "",
                )
                for step in row.trail or []
            ],
        )
    return answers


async def run_reflection(turn_id: str) -> int:
    """Run the policy's reflect lenses on a closed turn. Returns the number of answers added.

    A lease makes duplicates return 0 and gates every write.
    """
    from django_ai_sdk.oversight.models import Turn

    lease = await claim_for_reflect(turn_id)
    if lease is None:
        return 0
    try:
        turn = await Turn.objects.select_related("content").filter(id=turn_id).afirst()
        if turn is None:
            return 0
        judged = await judge_stored_turn(turn)
        if judged is None:
            await release_reflect_claim(turn_id, lease)
            return 0
        policy, context, answered, matched, found, masks = judged
        if turn.reflect_attempts < REFLECT_ATTEMPTS and any(item.needs_retry for item in answered):
            await release_reflect_claim(turn_id, lease)
            return 0
        kept = await sync_to_async(commit_reflection)(
            turn, policy, context, answered, matched, found, masks, lease=lease
        )
    except BaseException:
        await release_reflect_claim(turn_id, lease)
        raise
    return len(answered) if kept else 0


def claimable_q(now: datetime, pending_before: datetime | None = None) -> Q:
    from django_ai_sdk.oversight.models import Turn

    state = Turn.ReflectState
    pending = {"created_at__lt": pending_before} if pending_before is not None else {}
    return (
        Q(reflect_state=state.PENDING, **pending)
        | Q(reflect_state=state.RUNNING, reflect_lease_until__lt=now)
    ) & Q(reflect_attempts__lt=REFLECT_ATTEMPTS)


async def claim_for_reflect(turn_id: str) -> Any:
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


async def release_reflect_claim(turn_id: str, lease: Any) -> None:
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


def commit_reflection(
    turn: Turn,
    policy: ResolvedPolicy,
    context: TurnContext,
    answered: list[Answered],
    matched: list[Matched],
    found: dict[str, str],
    masks: list[MatchedMaskGroup],
    *,
    lease: Any,
) -> bool:
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
        record_answers(turn, policy, answered, matched, context=context, found=found)
        events = apply_reflect_actions(turn, context, matched, masks)
        Turn.objects.filter(pk=turn.pk).update(
            reflect_state=state.DONE, reflect_lease_until=None, reflected_at=timezone.now()
        )
        if events:
            from django_ai_sdk.oversight import tasks

            transaction.on_commit(lambda: tasks.enqueue_notifications(events))
    return True


def mark_reflect_failed(turns: QuerySet) -> int:
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


async def judge_stored_turn(
    turn: Turn,
) -> (
    tuple[
        ResolvedPolicy,
        TurnContext,
        list[Answered],
        list[Matched],
        dict[str, str],
        list[MatchedMaskGroup],
    ]
    | None
):
    from django_ai_sdk.oversight.models import LensAnswer, RuleMatch

    policy = await get_resolved_policy(turn.policy)
    if policy is None:
        return None
    content = await sync_to_async(load_content)(turn)
    context = context_from_turn(turn, content)
    found: dict[str, str] = {}
    if content is not None and turn.content is not None and turn.content.capture == "masked":
        masked = (context.input, context.output)
        context.input = await _stored_text(turn.user_message_id) or context.input
        context.output = await _stored_text(turn.message_id) or context.output
        found = {
            **unmasked_literals(masked[0], context.input),
            **unmasked_literals(masked[1], context.output),
        }
    existing = [row async for row in LensAnswer.objects.filter(turn=turn)]
    context.answers = answers_from(existing, policy.lenses)
    has_text = content is not None and "input" in content and "output" in content
    if not has_text:
        logger.info("Turn {} kept no text; reflect judges only what it kept", turn.id)
    retry = {
        row.lens
        for row in existing
        if should_retry(row.error, [(step or {}).get("error", "") for step in row.trail or []])
    }
    done = {
        rule async for rule in RuleMatch.objects.filter(turn=turn).values_list("rule", flat=True)
    }
    answered, matched, masks = await judge_reflect_stage(
        policy,
        context,
        retry,
        done,
        outcome=turn.outcome,
        content=content,
        has_text=has_text,
    )
    return policy, context, answered, matched, found, masks


async def judge_reflect_stage(
    policy: ResolvedPolicy,
    context: TurnContext,
    retry: Iterable[str],
    done: set[str],
    *,
    outcome: str,
    content: dict[str, Any] | None = None,
    has_text: bool = True,
    sample: bool = True,
) -> tuple[list[Answered], list[Matched], list[MatchedMaskGroup]]:
    has_answer = bool(context.output.strip()) or outcome != Outcome.BLOCKED
    lenses, late = reflect_lenses(
        policy, retry, answered=has_answer, content=content, has_text=has_text
    )
    with unguarded():
        answered = await answer_lenses(lenses, context, Stage.REFLECT, sample=sample)
        answered += await answer_lenses(late, context, Stage.REFLECT, sample=False)
    for item in answered:
        context.answers[item.lens.name] = item.answer
    failed = {name for name, answer in context.answers.items() if answer.error}
    failed |= {item.lens.name for item in answered if item.errored}
    matched = match_rules(
        policy,
        context.answers,
        Stage.REFLECT,
        done=done,
        errored=failed,
        fail_closed=policy.fails_closed,
    )
    masks = _reflect_masks(context, answered, matched, closed=policy.fails_closed)
    return answered, matched, masks


def reflect_lenses(
    policy: ResolvedPolicy,
    retry: Iterable[str],
    *,
    answered: bool,
    content: dict[str, Any] | None = None,
    has_text: bool = True,
) -> tuple[list[Lens], list[Lens]]:
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
    context: TurnContext, answered: list[Answered], matched: list[Matched], *, closed: bool
) -> list[MatchedMaskGroup]:
    groups, unpointed = collect_masks(
        matched, answered, subject=Subject.ANSWER, context=context, closed=closed
    )
    whole = [TextSpan(start=0, end=len(context.output), label="withheld")]
    for match, mask in unpointed:
        if context.output:
            groups.append((match, (whole, mask.template)))
        else:
            match.acted.remove(mask)
    return groups


async def _stored_text(message_id: str) -> str | None:
    messages = _messages(message_id)
    message = await messages.afirst() if messages is not None else None
    result = message.result if message is not None else None
    content = result.get("content") if isinstance(result, dict) else None
    return content if isinstance(content, str) else None


def _messages(message_id: str, *, lock: bool = False) -> QuerySet | None:
    from django_ai_sdk.conversation.models import Message

    if (pk := as_uuid(message_id)) is None:
        return None
    messages = Message.objects.filter(id=pk)
    return messages.select_for_update() if lock else messages


def _judgeable(
    lens: Lens, content: dict[str, Any] | None, has_text: bool, *, answered: bool = True
) -> bool:
    if lens.subject is Subject.TOOL_CALL:
        return False
    if not answered and lens.subject in {Subject.ANSWER, Subject.EXCHANGE}:
        return False
    if has_text:
        return True
    return lens.subject is Subject.SOURCES and bool((content or {}).get("sources"))


def apply_reflect_actions(
    turn: Turn, context: TurnContext, matched: list[Matched], groups: list[MatchedMaskGroup]
) -> list[tuple[dict[str, Any], list[str]]]:
    masks = [group for _, group in groups]
    rules = [match.rule.name for match, _ in groups]
    events = notify_events(
        turn.id,
        matched,
        thread_id=turn.thread_id,
        agent=turn.agent_name,
        channel=turn.channel,
        stage=str(Stage.REFLECT),
    )
    notes = rule_notes(matched, set())
    if (masks or notes) and turn.message_id:
        _update_stored_message(turn, context.output, masks, rules, notes)
    return events


def _update_stored_message(
    turn: Turn,
    judged: str,
    masks: list[tuple[list[Any], str]],
    rules: list[str],
    notes: list[dict[str, Any]],
) -> None:
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
