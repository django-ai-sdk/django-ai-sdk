"""Review queues: list waiting items and close one.

A reviewer says whether the rule that queued the turn was right (`agree`), or
dismisses the item, with an optional note. The item is claimed with a conditional
update, so if two reviewers close one item the second gets a ConflictError.
Nobody reviews a turn of their own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from asgiref.sync import sync_to_async
from django.utils import timezone

from django_ai_sdk.errors import UserError
from django_ai_sdk.oversight.content import load_contents
from django_ai_sdk.oversight.schemas import Page, ResolveIn, ReviewOut, ReviewStatus
from django_ai_sdk.oversight.services.access import (
    CONTENT,
    METRICS,
    REVIEW,
    OversightAccess,
    count_by,
    reader_id,
)
from django_ai_sdk.oversight.services.paging import keyset_page, page_size
from django_ai_sdk.oversight.services.turns import review_out, rows_of
from django_ai_sdk.permissions import ConflictError, PermissionDenied
from django_ai_sdk.utils import as_uuid

if TYPE_CHECKING:
    from django_ai_sdk.oversight.models import Turn
    from django_ai_sdk.types import UserType


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
        """List review items, oldest first.

        The turn's row is included only for readers with VIEW_OVERSIGHT_METRICS.
        Its content, and the reviewer's note, only where the reader may see it.
        """
        from django_ai_sdk.oversight.models import ReviewItem, Turn

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
        items, next_cursor = await keyset_page(qs, ("created_at", "pk"), limit, cursor)
        turns = list({r.turn_id: r.turn for r in items}.values())
        shown = await cls.content_ids(user, [t.id for t in turns], allowed=may_content)
        rows = {}
        if with_rows:
            # A turn's row is metrics: only for the turns in the reader's METRICS scope.
            metric = Turn.objects.filter(pk__in=[t.id for t in turns])
            in_scope = {
                pk
                async for pk in (await cls.scoped(user, METRICS, metric)).values_list(
                    "pk", flat=True
                )
            }
            listed = [t for t in turns if t.id in in_scope]
            contents = await sync_to_async(load_contents)([t for t in listed if t.id in shown])
            rows = {row.id: row for row in await rows_of(listed, shown, contents=contents)}
        return Page[ReviewOut](
            items=[
                review_out(r, rows.get(str(r.turn_id)), shown=r.turn_id in shown) for r in items
            ],
            limit=limit,
            next_cursor=next_cursor,
        )

    @classmethod
    async def queues(cls, *, user: UserType) -> dict[str, int]:
        """Count open review items per queue.

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
        """Close a review item with the reviewer's verdict, or dismiss it.

        Returns None for an item outside the reader's scope. Raises ConflictError
        if anyone already closed it.
        """
        from django_ai_sdk.oversight.models import ReviewItem

        await cls.has_perms(user, REVIEW)
        if answer.dismiss == (answer.agree is not None):
            raise UserError("Say whether the rule was right (agree), or dismiss the item.")
        scoped = await cls.scoped(user, REVIEW, ReviewItem.objects.select_related("turn"))
        pk = as_uuid(item_id)
        item = await scoped.filter(pk=pk).afirst() if pk else None
        if item is None:
            return None
        await _refuse_own(item.turn, user)
        closed = {
            "status": ReviewItem.Status.DISMISSED if answer.dismiss else ReviewItem.Status.DONE,
            "resolved_at": timezone.now(),
            "resolved_by_id": reader_id(user),
            "agree": answer.agree,
            "note": answer.note,
        }
        claimed = await ReviewItem.objects.filter(
            pk=item.pk, status=ReviewItem.Status.OPEN
        ).aupdate(**closed)
        if not claimed:
            raise ConflictError("This review is closed already.")
        for field, value in closed.items():
            setattr(item, field, value)
        shown = item.turn_id in await cls.content_ids(
            user, [item.turn_id], allowed=await cls.has_perms(user, CONTENT, raise_on_deny=False)
        )
        return review_out(item, shown=shown)


async def _refuse_own(turn: Turn, user: UserType) -> None:
    """Raise PermissionDenied when `user` asked this turn. Nobody reviews their own turn.

    A subagent's turn may carry no actor, so the turn that delegated it counts too.
    """
    from django_ai_sdk.oversight.models import Turn

    actors = {turn.actor_id}
    if turn.parent_turn_id:
        parent = Turn.objects.filter(pk=turn.parent_turn_id).values_list("actor_id", flat=True)
        actors.add(await parent.afirst())
    me = reader_id(user)
    if me and me in actors:
        raise PermissionDenied("Nobody reviews a turn of their own.")


__all__ = ["ReviewActions"]
