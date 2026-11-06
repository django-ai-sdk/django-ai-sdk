"""Delete expired oversight data in batches, then record that the prune ran.

The oversight manual's Retention section lists what goes. Custody hooks run in each
batch's transaction, and each batch is its own transaction, so a backlog never holds
one long lock.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from django.db import transaction
from django.utils import timezone

if TYPE_CHECKING:
    from collections.abc import Callable

    from django.db.models import QuerySet

# Rows deleted per transaction.
BATCH_SIZE = 500
# The reader label on the prune's access record.
PRUNER = "oversight_prune"


def prune(*, now: datetime | None = None, batch_size: int = BATCH_SIZE) -> dict[str, int]:
    """Delete what expired by `now`. Returns counts of content rows, turns and access records."""
    from django_ai_sdk.oversight.conf import setting, validate_retention
    from django_ai_sdk.oversight.models import AccessRecord, Turn, TurnContent
    from django_ai_sdk.oversight.services.access import write_access

    now = now or timezone.now()
    batch_size = max(1, batch_size)
    # A bad setting raises here: a metadata retention of 0 days would delete every access record.
    days = validate_retention(setting("RETENTION"))["metadata"]
    counts = {
        "content": _in_batches(
            TurnContent.objects.filter(expires_at__lte=now), batch_size, _delete_content
        ),
        "turns": _in_batches(Turn.objects.filter(expires_at__lte=now), batch_size, _delete_turns),
        "access": _in_batches(
            AccessRecord.objects.filter(created_at__lt=now - timedelta(days=days)),
            batch_size,
            _delete_access,
        ),
    }
    with transaction.atomic():
        write_access(
            None,
            AccessRecord.Operation.PRUNE,
            label=PRUNER,
            query={**counts, "expired_by": now.isoformat()},
        )
    return counts


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
    from django_ai_sdk.oversight.models import TurnContent

    # The turns stay. Their content link is set to NULL.
    return _delete_rows(TurnContent, ids)


def _delete_turns(ids: list[Any]) -> int:
    from django_ai_sdk.oversight import custody
    from django_ai_sdk.oversight.models import Turn, TurnContent

    custody.get().pruning(ids)
    contents = list(
        Turn.objects.filter(pk__in=ids, content__isnull=False).values_list("content_id", flat=True)
    )
    # Answers, matches and review items cascade with their turn.
    deleted = _delete_rows(Turn, ids)
    TurnContent.objects.filter(pk__in=contents).delete()
    return deleted


def _delete_access(ids: list[Any]) -> int:
    from django_ai_sdk.oversight.models import AccessRecord

    return _delete_rows(AccessRecord, ids)


def _delete_rows(model: Any, ids: list[Any]) -> int:
    """Delete `model` rows by pk. Returns how many of `model` went, not counting cascades."""
    _, by_model = model.objects.filter(pk__in=ids).delete()
    return by_model.get(model._meta.label, 0)


__all__ = ["BATCH_SIZE", "prune"]
