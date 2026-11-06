"""Review queues: list waiting items and resolve one.

Resolving saves the reviewer's answer to the item's lens (or rule) the same way
reflect saves one: a LensAnswer with the reviewer's user id, any rules it now
matches, and the note stored with the turn's content. The item is claimed with a
conditional update inside the same transaction. If two reviewers close one item,
the second gets a ConflictError.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from asgiref.sync import sync_to_async
from django.utils import timezone

from django_ai_sdk.errors import UserError
from django_ai_sdk.oversight.content import load_contents, stored_value
from django_ai_sdk.oversight.schemas import Page, ResolveIn, ReviewOut, ReviewStatus
from django_ai_sdk.oversight.services.access import (
    CONTENT,
    METRICS,
    REVIEW,
    OversightAccess,
    count_by,
    reader_id,
)
from django_ai_sdk.oversight.services.paging import cursor_query, keyset_page, page_size
from django_ai_sdk.oversight.services.turns import review_notes, review_out, rows_of
from django_ai_sdk.oversight.types import RULE_VERDICT_PREFIX, Answer, Stage
from django_ai_sdk.permissions import ConflictError, PermissionDenied
from django_ai_sdk.utils import as_uuid

if TYPE_CHECKING:
    from django_ai_sdk.oversight.guard.asking import Answered, Matched
    from django_ai_sdk.oversight.models import ReviewItem, Turn
    from django_ai_sdk.oversight.policy import ResolvedPolicy
    from django_ai_sdk.types import UserType

# The `answerer` value stored on a reviewer's answer.
REVIEWER = "reviewer"


class ReviewActions(OversightAccess):
    @classmethod
    async def list_reviews(
        cls,
        *,
        user: UserType,
        queue: str = "",
        status: ReviewStatus = "open",
        limit: int = 50,
        cursor: str = "",
    ) -> Page[ReviewOut]:
        """List review items, highest priority first, then oldest.

        The turn's row is included only for readers with VIEW_OVERSIGHT_METRICS.
        Its content is included only where the reader may see it.
        """
        from django_ai_sdk.oversight.models import AccessRecord, ReviewItem

        await cls.has_perms(user, REVIEW)
        limit = page_size(limit)
        with_rows = await cls.has_perms(user, METRICS, raise_on_deny=False)
        may_content = await cls.has_perms(user, CONTENT, raise_on_deny=False)
        related = ("turn", "turn__content") if may_content else ("turn",)
        qs = await cls.scoped(user, REVIEW, ReviewItem.objects.select_related(*related))
        if queue:
            qs = qs.filter(queue=queue)
        if status:
            qs = qs.filter(status=status)
        items, next_cursor = await keyset_page(qs, ("-priority", "created_at", "pk"), limit, cursor)
        turns = list({r.turn_id: r.turn for r in items}.values())
        shown = await cls.content_ids(user, [t.id for t in turns], allowed=may_content)
        contents = await sync_to_async(load_contents)([t for t in turns if t.id in shown])
        rows = (
            {row.id: row for row in await rows_of(turns, shown, contents=contents)}
            if with_rows
            else {}
        )
        await cls.record(
            user,
            AccessRecord.Operation.REVIEWS,
            turn_ids=[t.id for t in turns],
            content_ids=shown,
            query={"queue": queue, "status": status, **cursor_query(cursor)},
        )
        return Page[ReviewOut](
            items=[
                review_out(
                    r,
                    rows.get(str(r.turn_id)),
                    notes=review_notes(r.turn, contents.get(r.turn_id))
                    if r.turn_id in shown
                    else None,
                )
                for r in items
            ],
            limit=limit,
            next_cursor=next_cursor,
        )

    @classmethod
    async def queues(cls, *, user: UserType) -> dict[str, int]:
        """Count open review items per queue. Counts write no access record.

        `summarize` counts the same under VIEW_OVERSIGHT_METRICS. This one is for
        reviewers, who may lack that permission.
        """
        from django_ai_sdk.oversight.models import ReviewItem

        await cls.has_perms(user, REVIEW)
        qs = await cls.scoped(
            user, REVIEW, ReviewItem.objects.filter(status=ReviewItem.Status.OPEN)
        )
        return await count_by(qs, "queue")

    @classmethod
    async def resolve_review(
        cls, item_id: str, answer: ResolveIn, *, user: UserType
    ) -> ReviewOut | None:
        """Close a review item with a person's answer, or dismiss it (`answer.dismiss`).

        Returns None for an item outside the reader's scope. Raises ConflictError
        if anyone already closed it.
        """
        from django_ai_sdk.oversight.models import AccessRecord, ReviewItem

        await cls.has_perms(user, REVIEW)
        scoped = await cls.scoped(
            user, REVIEW, ReviewItem.objects.select_related("turn", "turn__content")
        )
        pk = as_uuid(item_id)
        item = await scoped.filter(pk=pk).afirst() if pk else None
        if item is None:
            return None
        if item.status != ReviewItem.Status.OPEN:
            raise ConflictError("This review is closed already.")
        _refuse_own(item.turn, user)
        dismiss = answer.dismiss
        if not dismiss:
            _refuse_open(item.turn)
        resolved, answered, matched = None, [], []
        name = (
            answer.lens or item.lens or (f"{RULE_VERDICT_PREFIX}{item.rule}" if item.rule else "")
        )
        if not dismiss and name:
            resolved, answered, matched = await _reviewer_answer(item.turn, name, answer)
        given = answered[0] if answered else None
        item.status = ReviewItem.Status.DISMISSED if dismiss else ReviewItem.Status.DONE
        item.resolved_at = timezone.now()
        item.resolution = {
            "by": reader_id(user),
            # Store the value as the record does, so names a reviewer typed stay out of it.
            "value": stored_value(given.lens, given.answer) if given else None,
            "agree": answer.agree,
        }
        written = await sync_to_async(_save_review_answer)(
            item, resolved, answered, matched, reader_id(user), note=answer.note
        )
        if written is None:
            raise ConflictError("This review is closed already.")
        if not dismiss:
            await sync_to_async(_notify_matched)(item.turn, matched)
        await cls.record(
            user,
            AccessRecord.Operation.REVIEW,
            turn_ids=[item.turn_id],
            query={"item": str(item.id)},
        )
        return review_out(item, notes=written[1] if answer.note else None)


def _notify_matched(turn: Turn, matched: list[Matched]) -> None:
    """Queue notifications for enforced rules that a reviewer's answer made match.

    This mirrors reflect, so a reviewer marking a leak still sends an alert. The
    stage is "review" so the event id differs from reflect's for the same rule.
    A reviewer's answer never masks or adds a notice to the turn's answer.
    """
    from django_ai_sdk.oversight.notifiers import match_events
    from django_ai_sdk.oversight.persist import enqueue_notifications

    events = match_events(
        turn.id, matched, agent=turn.agent_name, channel=turn.channel, stage="review"
    )
    if events:
        enqueue_notifications(events)


def _refuse_own(turn: Turn, user: UserType) -> None:
    """Raise PermissionDenied when `user` asked this turn. Nobody reviews their own turn."""
    if turn.actor_id and turn.actor_id == reader_id(user):
        raise PermissionDenied("Nobody reviews a turn of their own.")


def _refuse_open(turn: Turn) -> None:
    """Refuse answers on an open turn, since its close could match the same rules."""
    if turn.status == turn.Status.OPEN:
        raise ConflictError("This turn is still running; answer it once it closes.")


async def _reviewer_answer(
    turn: Turn, lens_name: str, given: ResolveIn
) -> tuple[ResolvedPolicy | None, list[Answered], list[Matched]]:
    """Check a person's answer against the turn and find the rules it now matches.

    `lens_name` is a lens of the turn's policy, or "rule:<name>" to judge a rule
    that matched the turn. Anything else raises UserError.
    """
    from django_ai_sdk.oversight.guard.asking import Answered, evaluate_rules
    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.models import LensAnswer, RuleMatch
    from django_ai_sdk.oversight.reflect import answers_from
    from django_ai_sdk.oversight.resolve import aget_policy

    resolved = await aget_policy(turn.policy)
    done = {r async for r in RuleMatch.objects.filter(turn=turn).values_list("rule", flat=True)}
    if lens_name.startswith(RULE_VERDICT_PREFIX):
        rule = lens_name.removeprefix(RULE_VERDICT_PREFIX)
        if rule not in done:
            raise UserError(f"Rule {rule!r} did not match this turn.")
        # Judging a rule is a yes/no answer (a Flag, the default) on a lens of its own.
        lens = Lens(name=lens_name)
        value = given.agree if given.agree is not None else given.value
    else:
        lens = resolved.lenses.get(lens_name) if resolved else None
        if lens is None:
            raise UserError(f"{lens_name!r} is not a lens of policy {turn.policy!r}.")
        value = given.value
    try:
        value = lens.answer.parse(lens, value)
    except (TypeError, ValueError) as exc:
        raise UserError(f"Not a valid answer for {lens_name!r}: {exc}") from exc
    answer = Answer(value=value, confidence=1.0, rationale=given.note, answerer=REVIEWER)
    matched = []
    if resolved is not None:
        answers = answers_from([a async for a in LensAnswer.objects.filter(turn=turn)])
        answers[lens_name] = answer
        matched = evaluate_rules(resolved, answers, Stage.REFLECT, done=done)
    return resolved, [Answered(lens, Stage.REFLECT, answer)], matched


def _save_review_answer(
    item: ReviewItem,
    resolved: ResolvedPolicy | None,
    answered: list[Answered],
    matched: list[Matched],
    reviewer: str,
    *,
    note: str = "",
) -> tuple[dict[str, Any], dict[str, str]] | None:
    """Claim and close `item` and save a person's answer on its turn, atomically.

    `note` is stored with the content under the item's id. Returns the result of
    `persist.save_answers`, or None (and writes nothing) if someone closed the
    item first.
    """
    from django.db import transaction

    from django_ai_sdk.oversight.models import ReviewItem
    from django_ai_sdk.oversight.persist import save_answers

    with transaction.atomic():
        claimed = ReviewItem.objects.filter(pk=item.pk, status=ReviewItem.Status.OPEN).update(
            status=item.status, resolved_at=item.resolved_at, resolution=item.resolution
        )
        if not claimed:
            return None
        notes = {str(item.pk): note} if note else None
        return save_answers(item.turn, resolved, answered, matched, reviewer=reviewer, notes=notes)


__all__ = ["ReviewActions"]
