"""Write a turn's row, answers, matches and content.

The guard calls `persist_turn` when its gate ends and again at close. Reflect and
reviewers add answers later through `save_answers`. The rows keep values, labels,
scores, counts and errors. Anything that can quote the turn goes into its content.
"""

from __future__ import annotations

import time
import uuid
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from django.utils import timezone

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.actions import Alert, Notify, Queue
from django_ai_sdk.oversight.content import (
    Masker,
    add_to_content,
    capture_content,
    found_literals,
    masker_for_capture,
    stored_value,
    turn_found,
    without_nul,
)
from django_ai_sdk.oversight.guard.asking import should_retry
from django_ai_sdk.oversight.types import (
    RULE_VERDICT_PREFIX,
    Answer,
    Capture,
    Outcome,
    Severity,
    Stage,
    Subject,
    TurnContext,
    most_severe,
)
from django_ai_sdk.utils import as_uuid

if TYPE_CHECKING:
    from collections.abc import Iterable

    from django_ai_sdk.oversight.guard import TurnGuard
    from django_ai_sdk.oversight.guard.asking import Answered, Matched
    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.models import LensAnswer, Turn
    from django_ai_sdk.oversight.policy import ResolvedPolicy

logger = get_logger(__name__)


def persist_turn(guard: TurnGuard, *, closing: bool = True) -> str:
    """Write the turn row and any answers and matches not written yet.

    Called twice per turn, each time in one transaction. When the gate ends
    (`closing=False`) the row is inserted as OPEN with the gate's answers and
    matches. A turn that never closes still leaves a row for `oversight_sweep` to
    mark INTERRUPTED. At close the row is updated (or inserted if missing), the
    remaining answers and matches are added, and the content is written.

    Content is stored through custody. A forgotten scope stores no new content and
    no judge text. Written rows go to `custody.chained`, and if it raises the whole
    write is rolled back.
    """
    from django.db import transaction

    from django_ai_sdk.oversight import custody
    from django_ai_sdk.oversight.models import Turn, TurnContent

    policy = guard.resolved
    now = timezone.now()
    created = guard.created_at or now
    saved_answered, saved_matched = len(guard._answer_ids), guard._matched_saved
    content: dict[str, Any] | None = None
    keep: Masker | None = None
    if closing:
        found = turn_found(guard) if policy.capture is Capture.MASKED else None
        keep = masker_for_capture(policy.capture, found)
        # PostgreSQL rejects NUL characters in text, so strip them.
        content = without_nul(capture_content(guard, policy.capture, keep))
    # Answer texts go with the content. Rows written earlier keep their ids.
    answer_rows, texts = _build_answers(guard.answered, keep=keep, ids=guard._answer_ids)
    texts = without_nul(texts)
    added = answer_rows[saved_answered:]
    matched = guard.matched[saved_matched:]
    scope = custody.scope_for(guard.thread_id, guard.id)
    custodian = custody.get()
    columns = _turn_columns(guard, closing=closing, created=created)

    with transaction.atomic():
        row = None
        if content is not None:
            # Judge text is stored only inside the content, never on its own.
            stored = None
            try:
                stored = custodian.store(scope, {**content, "texts": texts} if texts else content)
            except custody.Forgotten:
                pass
            if stored is not None:
                row = TurnContent.objects.create(
                    scope=scope,
                    stored=stored,
                    capture=str(policy.capture),
                    expires_at=now + timedelta(days=policy.retention["content"]),
                )
        if closing:
            columns["content"] = row
        # The actor is written once, so a forget while the turn was open stays done.
        updated = {key: value for key, value in columns.items() if key != "actor_id"}
        written = guard._written and Turn.objects.filter(pk=guard.id).update(**updated)
        if written:
            # Lock and re-read, since a reviewer may have changed labels or severity since.
            turn = Turn.objects.select_for_update().get(pk=guard.id)
        else:
            turn = Turn.objects.create(id=guard.id, created_at=created, **columns)
        _record_judgments(
            turn,
            turn,
            added,
            guard.answered[saved_answered:],
            matched,
            floor=Severity.HIGH if guard.outcome is Outcome.BLOCKED else "",
            head=[("turn", turn)],
        )
        if closing and columns.get("reflect_state") == Turn.ReflectState.PENDING:
            transaction.on_commit(lambda: enqueue_reflect(turn.id))
        if events := guard.notifications(matched):
            transaction.on_commit(lambda: enqueue_notifications(events))
    guard._written = True
    guard._answer_ids = [str(row.id) for row in answer_rows]
    guard._matched_saved = len(guard.matched)
    return str(turn.id)


def _turn_columns(guard: TurnGuard, *, closing: bool, created: Any) -> dict[str, Any]:
    """Build the turn row's columns from the guard's current state."""
    from django_ai_sdk.oversight.models import Turn

    policy = guard.resolved
    columns: dict[str, Any] = {
        "status": Turn.Status.CLOSED if closing else Turn.Status.OPEN,
        "expires_at": created + timedelta(days=policy.retention["metadata"]),
        "channel": fit_column(Turn, "channel", guard.channel),
        "agent_id": fit_column(Turn, "agent_id", guard.context.agent_id),
        "agent_name": fit_column(Turn, "agent_name", guard.context.agent_name),
        "agent_fingerprint": guard.agent_fingerprint,
        "policy": policy.name,
        "thread_id": guard.thread_id,
        "user_message_id": guard.user_message_id,
        "message_id": guard.message_id,
        "parent_turn_id": as_uuid(guard.parent_turn_id),
        "actor_id": fit_column(Turn, "actor_id", guard.actor_id),
        "model": fit_column(Turn, "model", guard.usage.model),
        "prompt_tokens": guard.usage.prompt_tokens,
        "completion_tokens": guard.usage.completion_tokens,
        "cost": _cost(guard),
        "latency_ms": round((time.perf_counter() - guard._started) * 1000, 1) if closing else None,
        "error": fit_column(Turn, "error", guard.error),
    }
    columns["outcome"], columns["report_outcome"] = _outcome_columns(guard, closing=closing)
    if closing and _needs_reflect(guard):
        columns["reflect_state"] = Turn.ReflectState.PENDING
    return columns


def _outcome_columns(guard: TurnGuard, *, closing: bool) -> tuple[str, str]:
    """Return the outcome and report outcome to store on the row.

    "allowed" is stored as blank while the turn is open or when its run failed
    (a crash, a disconnect, no result), since nothing got through. Outcomes from
    rules that acted are always kept.
    """
    ended = closing and not guard.run_error

    def column(outcome: Outcome) -> str:
        return "" if not ended and outcome is Outcome.ALLOWED else str(outcome)

    return column(guard.outcome), column(guard.report_outcome)


def _build_answers(
    answered: Iterable[Answered],
    *,
    keep: Masker | None,
    reviewer: str = "",
    ids: list[str] | None = None,
) -> tuple[list[LensAnswer], dict[str, dict[str, Any]]]:
    """Build unsaved LensAnswer rows and their content texts, keyed by row id.

    `keep` decides how texts are kept. None keeps no texts. `ids` are the ids of
    leading rows that were already written.
    """
    from django_ai_sdk.oversight.models import LensAnswer

    ids = ids or []
    answer_rows: list[LensAnswer] = []
    texts: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(answered):
        columns, kept = split_answer(item.lens, item.stage, item.answer, keep=keep)
        row = LensAnswer(reviewer=reviewer, **columns)
        if index < len(ids):
            row.id = uuid.UUID(ids[index])
        answer_rows.append(row)
        if kept:
            texts[str(row.id)] = kept
    return answer_rows, texts


def split_answer(
    lens: Lens, stage: Stage | str, answer: Answer, *, keep: Masker | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split an answer into LensAnswer columns and the parts that can quote the turn.

    `value` in the columns is what the record keeps and the client sees. The
    quoting parts go to the content through `keep`: the rationale, entity names,
    per-item scores, and the same for answers on the trail. None keeps none of them.
    """
    from django_ai_sdk.oversight.models import LensAnswer
    from django_ai_sdk.utils import serialize

    value = stored_value(lens, answer)
    trail = [_trail(lens, a) for a in answer.trail]
    columns = {
        "lens": lens.name,
        "stage": str(stage),
        "answerer": fit_column(LensAnswer, "answerer", answer.answerer),
        "model": fit_column(LensAnswer, "model", answer.model),
        "value": without_nul(value),
        "confidence": answer.confidence if answer.ok else None,
        "spans": without_nul([s.model_dump() for s in answer.spans]),
        "trail": without_nul(trail),
        "error": fit_column(LensAnswer, "error", answer.error),
        "cost": answer.cost,
        "latency_ms": answer.latency_ms,
    }
    if keep is None:
        return columns, {}
    kept: dict[str, Any] = {}
    said = serialize(answer.value)
    if said is not None and said != value:
        kept["value"] = keep(said)
    if answer.rationale:
        kept["rationale"] = keep.text(answer.rationale)
    if answer.items:
        kept["items"] = keep.keys(answer.items)
    said_trail = [serialize(a.value) for a in answer.trail]
    if said_trail != [step["value"] for step in trail]:
        kept["trail"] = keep(said_trail)
    return columns, kept


def fit_column(model: Any, name: str, value: str) -> str:
    """Strip NUL characters and truncate `value` to fit the column."""
    return value.replace("\x00", "")[: model._meta.get_field(name).max_length]


def _trail(lens: Lens, answer: Answer) -> dict[str, Any]:
    return {
        "answerer": answer.answerer,
        "model": answer.model,
        "value": stored_value(lens, answer),
        "confidence": answer.confidence,
        "error": answer.error,
        "cost": answer.cost,
        "latency_ms": answer.latency_ms,
    }


def _cost(guard: TurnGuard) -> float | None:
    from django_ai_sdk.pricing import cost_for

    return cost_for(guard.usage.model, guard.usage.prompt_tokens, guard.usage.completion_tokens)


def _needs_reflect(guard: TurnGuard) -> bool:
    unfinished = any(
        should_retry(item.answer.error, [step.error for step in item.answer.trail])
        for item in guard.answered
    )
    return unfinished or guard.resolved.has_stage(Stage.REFLECT)


def enqueue_notifications(events: list[tuple[dict[str, Any], list[str]]]) -> None:
    """Queue notifications for the worker, so a slow notifier cannot hold the stream open."""
    _safe_enqueue(
        "send_notifications",
        [[event, names] for event, names in events],
        what="oversight notifications",
    )


def enqueue_reflect(turn_id: Any) -> bool:
    """Queue reflection of `turn_id`. False when it could not be queued."""
    return _safe_enqueue("reflect_turn", str(turn_id), what=f"reflection for turn {turn_id}")


def _safe_enqueue(task_name: str, *args: Any, what: str) -> bool:
    """Queue the `oversight.tasks` task `task_name`. False, and logged, when that fails.

    A missing task backend must not fail the turn, so the error is only logged.
    The task is looked up by name because `oversight.tasks` imports this module.
    """
    try:
        from django_ai_sdk.oversight import tasks

        getattr(tasks, task_name).enqueue(*args)
    except Exception:
        logger.exception("Could not queue {}; is a task backend configured?", what)
        return False
    return True


def save_answers(
    turn: Turn,
    # None for a reviewer's answer when the turn's policy no longer exists.
    policy: ResolvedPolicy | None,
    answered: list[Answered],
    matched: list[Matched],
    *,
    reviewer: str = "",
    view: TurnContext | None = None,
    notes: dict[str, str] | None = None,
    found: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Add later answers and matches to a written turn and update its labels.

    Answer texts and reviewer `notes` (by review item id) go into the content, as
    at close. With MASKED capture they are masked using what the lenses found in
    `view`, the text reflect judged, and `found`, what the turn masked at close. Returns (texts by answer id, notes by item id)
    as stored, both empty when the turn keeps no text or its content is gone.

    The turn is locked first. Rules that already matched are removed from
    `matched` in place, because reflect and a reviewer may record at the same time
    and a rule matches a turn only once (PROD-9).
    """
    from django.db import transaction

    from django_ai_sdk.oversight.models import RuleMatch, Turn

    if view is not None:
        found = {**(found or {}), **found_literals(answered, {s: view.text(s) for s in Subject})}
    keep = masker_for_capture(turn.content.capture, found) if turn.content else None
    answer_rows, texts = _build_answers(answered, keep=keep, reviewer=reviewer)
    texts = without_nul(texts)
    kept_notes = (
        {key: keep.text(note) for key, note in (notes or {}).items() if note} if keep else {}
    )
    with transaction.atomic():
        # Lock before reading, since reflect and a reviewer may record at the same time.
        current = Turn.objects.select_for_update().only("labels", "severity").get(pk=turn.pk)
        if matched:
            done = set(RuleMatch.objects.filter(turn=turn).values_list("rule", flat=True))
            matched[:] = [m for m in matched if m.rule.name not in done]
        if (texts or kept_notes) and not add_to_content(turn, texts=texts, notes=kept_notes):
            texts, kept_notes = {}, {}
        if not answered and not matched:
            return texts, kept_notes
        _record_judgments(turn, current, answer_rows, answered, matched)
    return texts, kept_notes


def _record_judgments(
    turn: Turn,
    locked: Turn,
    rows: list[Any],
    answered: list[Answered],
    matched: list[Matched],
    *,
    floor: str = "",
    head: list[tuple[str, Any]] | None = None,
) -> None:
    """Save answers, matches and review items, merge labels and severity into the turn, chain them.

    Run inside a transaction. `rows` come from `_build_answers(answered)`. `locked`
    is the turn's row as read under the lock; it gets the merged labels and
    severity. `floor` is the least severity to record, `head` rows to chain first.
    """
    from django_ai_sdk.oversight import custody
    from django_ai_sdk.oversight.models import LensAnswer, ReviewItem, RuleMatch, Turn

    for row in rows:
        row.turn = turn
    answer_rows = LensAnswer.objects.bulk_create(rows)
    match_rows = RuleMatch.objects.bulk_create(
        RuleMatch(
            turn=turn,
            rule=match.rule.name,
            stage=str(match.stage),
            mode=str(match.rule.mode),
            actions=[a.to_dict() for a in match.rule.then],
            acted=bool(match.acted),
            condition=match.rule.when.to_dict(),
        )
        for match in matched
    )
    acted = [(match, action) for match in matched for action in match.acted]
    reviews = {(a.queue, "", m.rule.name) for m, a in acted if isinstance(a, Queue)}
    priority = {a.queue: a.priority for m, a in reversed(acted) if isinstance(a, Queue)}
    # A retried reflect or a second reviewer may request a review that already exists.
    ReviewItem.objects.bulk_create(
        [
            ReviewItem(
                turn=turn, queue=queue, lens=lens, rule=rule, priority=priority.get(queue, 0)
            )
            for queue, lens, rule in sorted(reviews)
        ],
        ignore_conflicts=True,
    )
    # A reviewer's verdict on a rule ("rule:<name>") is not a label.
    locked.labels = {
        **(locked.labels or {}),
        **{
            row.lens: row.value
            for item, row in zip(answered, answer_rows, strict=True)
            if item.answer.ok and not item.lens.name.startswith(RULE_VERDICT_PREFIX)
        },
    }
    alerts = (a.severity for m, a in acted if isinstance(a, Alert | Notify))
    locked.severity = most_severe(locked.severity, floor, *alerts)
    Turn.objects.filter(pk=turn.pk).update(labels=locked.labels, severity=locked.severity)
    chained = [("answer", row) for row in answer_rows] + [("match", row) for row in match_rows]
    custody.get().chained([*(head or []), *chained])


__all__ = [
    "enqueue_notifications",
    "enqueue_reflect",
    "fit_column",
    "persist_turn",
    "save_answers",
    "split_answer",
]
