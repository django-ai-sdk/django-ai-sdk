"""Who may read the oversight record, read its content, and answer its review queues."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from asgiref.sync import sync_to_async
from django.utils.module_loading import import_string

from django_ai_sdk.permissions import (
    DOMAIN_PERMISSION_DEFAULTS,
    BasePermission,
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


class OversightDefaultPermission(BasePermission):
    """A model permission per operation (never `is_staff`); override `scope_turns` to narrow."""

    MODEL_PERMISSIONS: dict[Operation, str] = {  # noqa: RUF012
        Operation.VIEW_OVERSIGHT_METRICS: "django_ai_sdk_oversight.view_turn",
        Operation.VIEW_OVERSIGHT_CONTENT: "django_ai_sdk_oversight.view_turncontent",
        Operation.REVIEW_OVERSIGHT: "django_ai_sdk_oversight.change_reviewitem",
    }

    async def has_permission(self, user: UserType, operation: Operation, **kwargs: Any) -> bool:
        from django_ai_sdk.oversight.apps import MODEL_PERMISSIONS as added

        perm = self.MODEL_PERMISSIONS.get(operation) or added.get(operation)
        ahas_perm = getattr(user, "ahas_perm", None)
        return perm is not None and ahas_perm is not None and await ahas_perm(perm)

    def get_queryset_perms(
        self, user: UserType, operation: Operation, queryset: QuerySet
    ) -> QuerySet:
        from django_ai_sdk.oversight.models import ReviewItem, Turn

        if queryset.model is Turn:
            return self.scope_turns(user, operation, queryset)
        if queryset.model is ReviewItem:
            return self.scope_reviews(user, operation, queryset)
        return queryset

    def scope_turns(self, user: UserType, operation: Operation, queryset: QuerySet) -> QuerySet:
        """Narrow the turns `user` can reach for `operation`. Default: all of them."""
        return queryset

    def scope_reviews(self, user: UserType, operation: Operation, queryset: QuerySet) -> QuerySet:
        from django_ai_sdk.oversight.models import Turn

        if type(self).scope_turns is OversightDefaultPermission.scope_turns:
            return queryset
        return queryset.filter(turn__in=self.scope_turns(user, operation, Turn.objects.all()))


class OversightAccess(PermissionsMixin):
    domain = PermissionDomain.OVERSIGHT

    @classmethod
    async def scoped(cls, user: UserType, operation: Operation, queryset: QuerySet) -> QuerySet:
        """In a thread, because a scope hook may query the database."""
        return await sync_to_async(cls.has_queryset_perms)(user, operation, queryset=queryset)

    @classmethod
    def get_default_permissions(cls) -> list[Any]:
        """An empty setting falls back; a record open to everyone says `[AllowAll]`."""
        return super().get_default_permissions() or [
            import_string(path) for path in DOMAIN_PERMISSION_DEFAULTS[cls.domain]
        ]

    @classmethod
    async def content_ids(
        cls, user: UserType, turn_ids: Iterable[Any], *, allowed: bool
    ) -> set[Any]:
        """The ids whose content `user` may see: the one content gate of every read.

        `allowed` is whether the caller may see CONTENT at all, so the permission
        chain runs once per read.
        """
        from django_ai_sdk.oversight.models import Turn

        ids = list(turn_ids)
        if not ids or not allowed:
            return set()
        visible = await cls.scoped(user, CONTENT, Turn.objects.filter(pk__in=ids))
        return {pk async for pk in visible.values_list("pk", flat=True)}


def reader_id(user: UserType) -> str:
    pk = user_pk(user)
    return "" if pk is None else str(pk)


__all__ = [
    "CONTENT",
    "METRICS",
    "REVIEW",
    "OversightAccess",
    "OversightDefaultPermission",
    "reader_id",
]
