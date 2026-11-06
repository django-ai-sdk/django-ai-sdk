"""Permission checks, scopes and access records shared by every review-API read."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

from asgiref.sync import sync_to_async
from django.utils import timezone

from django_ai_sdk.permissions import (
    Operation,
    PermissionDomain,
    PermissionsMixin,
    user_pk,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from django.db.models import QuerySet

    from django_ai_sdk.types import UserType

METRICS = Operation.VIEW_OVERSIGHT_METRICS
CONTENT = Operation.VIEW_OVERSIGHT_CONTENT
REVIEW = Operation.REVIEW_OVERSIGHT

# Identical reads by the same reader within this many seconds share one record.
REPEAT_SECONDS = 60


class OversightAccess(PermissionsMixin):
    """Checks the `oversight` permission domain and records each read."""

    domain = PermissionDomain.OVERSIGHT

    @classmethod
    async def scoped(cls, user: UserType, operation: Operation, queryset: QuerySet) -> QuerySet:
        """Narrow `queryset` to what `user` may reach for `operation`.

        Runs in a thread because a scope hook may query the database.
        """
        return await sync_to_async(cls.has_queryset_perms)(user, operation, queryset=queryset)

    @classmethod
    async def content_ids(
        cls, user: UserType, turn_ids: Iterable[Any], *, allowed: bool
    ) -> set[Any]:
        """Return the ids in `turn_ids` whose content `user` may see.

        `allowed` is whether the caller may see CONTENT at all, which it needs anyway, so the
        permission chain runs once per read.
        """
        from django_ai_sdk.oversight.models import Turn

        ids = list(turn_ids)
        if not ids or not allowed:
            return set()
        visible = await cls.scoped(user, CONTENT, Turn.objects.filter(pk__in=ids))
        return {pk async for pk in visible.values_list("pk", flat=True)}

    @classmethod
    async def record(
        cls,
        user: UserType,
        operation: str,
        *,
        turn_ids: Iterable[Any] = (),
        content_ids: Iterable[Any] = (),
        query: dict[str, Any] | None = None,
    ) -> Any:
        """Record who read which turns. A repeat within a minute reuses the last record.

        `content_ids` are the turns shown with content. When that is only some of
        `turn_ids`, they are stored in `query["content_ids"]` so the record shows
        exactly whose content was seen.
        """
        ids = [str(t) for t in turn_ids]
        query = dict(query or {})
        shown = {str(t) for t in content_ids}
        if shown and shown != set(ids):
            query["content_ids"] = [t for t in ids if t in shown]
        return await sync_to_async(_write_access)(user, operation, ids, bool(shown), query)


def reader_id(user: UserType) -> str:
    """The user's pk as stored on a record, or "" for anonymous."""
    pk = user_pk(user)
    return "" if pk is None else str(pk)


async def count_by(qs: QuerySet, field: str) -> dict[str, Any]:
    """Count the rows of `qs` per value of `field`."""
    from django.db.models import Count

    return {
        row[field]: row["n"] async for row in qs.order_by().values(field).annotate(n=Count("pk"))
    }


def reader_label(user: UserType) -> str:
    """Name the reader on the access record, so reads can be traced to a person."""
    pk = user_pk(user)
    if pk is None:
        return "anonymous"
    username = getattr(user, "get_username", lambda: "")()
    return f"{username} ({pk})" if username else str(pk)


def _write_access(
    user: UserType, operation: str, turn_ids: list[str], content: bool, query: dict[str, Any]
) -> Any:
    from django.db import transaction

    from django_ai_sdk.oversight.models import AccessRecord

    reader = reader_id(user)
    with transaction.atomic():
        last = (
            AccessRecord.objects.filter(
                reader_id=reader,
                operation=operation,
                created_at__gte=timezone.now() - timedelta(seconds=REPEAT_SECONDS),
            )
            .order_by("-created_at")
            .first()
        )
        if (
            last is not None
            and last.query == query
            and last.turn_ids == turn_ids
            and last.content == content
        ):
            return last
        return write_access(user, operation, turn_ids=turn_ids, content=content, query=query)


def write_access(
    user: UserType | None,
    operation: str,
    *,
    label: str = "",
    **fields: Any,
) -> Any:
    """Create an access record and hand it to custody to chain. Run in a transaction.

    `label` names a reader that is no user, such as a command.
    """
    from django_ai_sdk.oversight import custody
    from django_ai_sdk.oversight.models import AccessRecord

    record = AccessRecord.objects.create(
        reader_id=reader_id(user) if user is not None else "",
        reader_label=(label or reader_label(user))[:255],
        operation=operation,
        **fields,
    )
    custody.get().chained([("access", record)])
    return record


__all__ = [
    "CONTENT",
    "METRICS",
    "REVIEW",
    "OversightAccess",
    "count_by",
    "reader_id",
    "reader_label",
    "write_access",
]
