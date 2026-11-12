"""Delete expired oversight data in batches.

The oversight manual's Retention section lists what goes. Each batch is its own
transaction, so a backlog never holds one long lock.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django.db import transaction
from django.utils import timezone

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    from django.db.models import QuerySet

# Rows deleted per transaction.
BATCH_SIZE = 500


def prune(*, now: datetime | None = None, batch_size: int = BATCH_SIZE) -> dict[str, int]:
    """Delete what expired by `now`. Returns counts of content rows, turns and open reviews."""
    from django_ai_sdk.oversight.models import ReviewItem, Turn, TurnContent

    now = now or timezone.now()
    batch_size = max(1, batch_size)
    return {
        "content": _in_batches(
            TurnContent.objects.filter(expires_at__lte=now), batch_size, _delete_content
        ),
        # Retention wins over a waiting review; the record says how many went unreviewed.
        "open_reviews": ReviewItem.objects.filter(
            turn__expires_at__lte=now, status=ReviewItem.Status.OPEN
        ).count(),
        "turns": _in_batches(Turn.objects.filter(expires_at__lte=now), batch_size, _delete_turns),
    }


def _in_batches(qs: QuerySet, size: int, delete: Callable[[list[Any]], int]) -> int:
    """Delete the rows of `qs` in batches, one transaction per batch."""
    total = 0
    while True:
        ids = list(qs.order_by("pk").values_list("pk", flat=True)[:size])
        if not ids:
            return total
        with transaction.atomic():
            total += delete(ids)
        if len(ids) < size:
            return total


def _delete_content(ids: list[Any]) -> int:
    from django_ai_sdk.oversight.models import ReviewItem, TurnContent

    # The turns stay. Their content link is set to NULL, and reviewers' notes go with it.
    ReviewItem.objects.filter(turn__content__in=ids).exclude(note="").update(note="")
    return _delete_rows(TurnContent, ids)


def _delete_turns(ids: list[Any]) -> int:
    from django_ai_sdk.oversight.models import Turn, TurnContent

    contents = list(
        Turn.objects.filter(pk__in=ids, content__isnull=False).values_list("content_id", flat=True)
    )
    # Answers, matches and review items cascade with their turn.
    deleted = _delete_rows(Turn, ids)
    TurnContent.objects.filter(pk__in=contents).delete()
    return deleted


def _delete_rows(model: Any, ids: list[Any]) -> int:
    """Delete `model` rows by pk. Returns how many of `model` went, not counting cascades."""
    _, by_model = model.objects.filter(pk__in=ids).delete()
    return by_model.get(model._meta.label, 0)


__all__ = ["BATCH_SIZE", "prune"]
