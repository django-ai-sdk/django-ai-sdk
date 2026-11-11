"""Permission checks and scopes shared by every review-API read."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from asgiref.sync import sync_to_async

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


class OversightAccess(PermissionsMixin):
    """Checks the `oversight` permission domain."""

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


__all__ = [
    "CONTENT",
    "METRICS",
    "REVIEW",
    "OversightAccess",
    "count_by",
    "reader_id",
]
