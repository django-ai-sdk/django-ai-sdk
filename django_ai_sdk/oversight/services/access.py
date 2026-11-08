"""Permission checks, scopes and access records shared by every review-API read."""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import TYPE_CHECKING, Annotated, Any

from asgiref.sync import sync_to_async
from django.db import IntegrityError
from django.utils import timezone
from pydantic import Field

from django_ai_sdk.errors import UserError
from django_ai_sdk.permissions import (
    ConflictError,
    Operation,
    PermissionDomain,
    PermissionsMixin,
    user_pk,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from django.db.models import QuerySet

    from django_ai_sdk.types import UserType

METRICS = Operation.VIEW_OVERSIGHT_METRICS
CONTENT = Operation.VIEW_OVERSIGHT_CONTENT
REVIEW = Operation.REVIEW_OVERSIGHT

# Identical reads by the same reader within this many seconds share one record.
REPEAT_SECONDS = 60
# Tries of an operator's change when another change of the same target commits first.
ATTEMPTS = 3

# An operator's reason for a change, as a request field. Whether one is needed, and
# how long, is the service's rule (`require_reason`), so direct callers get it too.
Reason = Annotated[str, Field(max_length=2_000)]


def require_reason(reason: str | None, message: str, *, min_chars: int = 1) -> str:
    """The stripped reason. Raise UserError(message) when it is shorter than `min_chars`."""
    reason = (reason or "").strip()
    if len(reason) < min_chars:
        raise UserError(message)
    return reason


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
    def get_default_permissions(cls) -> list[Any]:
        """The domain's classes. An empty `AI_SDK_PERMISSIONS["oversight"]` falls back.

        As for agents: a record open to everyone says so with `[AllowAll]`.
        """
        from django.utils.module_loading import import_string

        from django_ai_sdk.permissions import DOMAIN_PERMISSION_DEFAULTS

        return super().get_default_permissions() or [
            import_string(path) for path in DOMAIN_PERMISSION_DEFAULTS[cls.domain]
        ]

    @classmethod
    async def content_ids(
        cls, user: UserType, turn_ids: Iterable[Any], *, allowed: bool
    ) -> set[Any]:
        """Return the ids in `turn_ids` whose content `user` may see.

        The one content gate of every read, a list or one turn: the CONTENT scope
        (`get_queryset_perms`), never a per-object check. `allowed` is whether the
        caller may see CONTENT at all, which it needs anyway, so the permission chain
        runs once per read.
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
    chained: Iterable[tuple[str, Any]] = (),
    label: str = "",
    **fields: Any,
) -> Any:
    """Create an access record and chain it after the `chained` rows. Run in a transaction.

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
    custody.get().chained([*chained, ("access", record)])
    return record


# Operator-change events get their id from the access record, so a resend is the same event.
CHANGE_EVENTS = uuid.UUID("0f7c1b8e-5d35-4a8e-9a43-2a0c6f1e7d21")


async def append_change(
    model: Any,
    kind: str,
    key: dict[str, Any],
    fields: dict[str, Any],
    *,
    user: UserType,
    operation: str,
    describe: Callable[[Any], tuple[dict[str, Any], dict[str, Any] | None]],
    conflict: str,
) -> Any:
    """Append the next version of an `OperatorChange` row for `key`, and record it.

    `describe(before)` gets the row in force, or None, and returns the access record's
    query and the event to send after commit (None for none). It may raise UserError.
    The row and the record are chained as `kind` and "access". When another change of
    `key` commits first, it tries again, and after ATTEMPTS tries raises
    ConflictError(conflict).
    """
    from django.db import transaction

    from django_ai_sdk.oversight.persist import enqueue_notifications

    def write() -> Any:
        with transaction.atomic():
            before = model.objects.history(**key).first()
            query, event = describe(before)
            row = model.objects.create(
                **key,
                **fields,
                version=before.version + 1 if before else 1,
                changed_by_id=reader_id(user),
            )
            record = write_access(user, operation, chained=[(kind, row)], query=query)
            if event is not None:
                sent = {
                    "id": str(uuid.uuid5(CHANGE_EVENTS, str(record.id))),
                    **event,
                    "by": reader_label(user),
                    "at": timezone.now().isoformat(),
                }
                transaction.on_commit(lambda: enqueue_notifications([(sent, [])]))
        return row

    for attempt in range(ATTEMPTS):
        try:
            return await sync_to_async(write)()
        except IntegrityError:
            if attempt == ATTEMPTS - 1:
                raise ConflictError(conflict) from None
    # The loop returns or raises; the type checker cannot see that.
    raise AssertionError("unreachable")


__all__ = [
    "ATTEMPTS",
    "CONTENT",
    "METRICS",
    "REVIEW",
    "OversightAccess",
    "Reason",
    "append_change",
    "count_by",
    "reader_id",
    "reader_label",
    "require_reason",
    "write_access",
]
