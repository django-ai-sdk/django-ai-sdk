"""Who may read the oversight record, read its content, and answer its review queues.

The default for the `oversight` permission domain: each operation needs the Django
model permission in `MODEL_PERMISSIONS`, or the one an opt-in app added
(`oversight.apps.extend`), never `is_staff`. Subclass it and override
`scope_turns`/`scope_reviews` to narrow which turns a reader reaches (the oversight
manual's Review API).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from asgiref.sync import sync_to_async

from django_ai_sdk.permissions import (
    BasePermission,
    Operation,
    PermissionDomain,
    PermissionsMixin,
    ahas_model_perm,
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
    """Check a model permission per operation, and narrow turns with the scope hooks."""

    MODEL_PERMISSIONS: dict[Operation, str] = {  # noqa: RUF012
        Operation.VIEW_OVERSIGHT_METRICS: "django_ai_sdk_oversight.view_turn",
        Operation.VIEW_OVERSIGHT_CONTENT: "django_ai_sdk_oversight.view_turncontent",
        Operation.REVIEW_OVERSIGHT: "django_ai_sdk_oversight.change_reviewitem",
    }

    async def has_permission(self, user: UserType, operation: Operation, **kwargs: Any) -> bool:
        from django_ai_sdk.oversight.apps import MODEL_PERMISSIONS as added

        perm = self.MODEL_PERMISSIONS.get(operation) or added.get(operation)
        return perm is not None and await ahas_model_perm(user, perm)

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
        """Narrow review items to those whose turn `scope_turns` keeps."""
        from django_ai_sdk.oversight.models import Turn

        if type(self).scope_turns is OversightDefaultPermission.scope_turns:
            return queryset
        return queryset.filter(turn__in=self.scope_turns(user, operation, Turn.objects.all()))


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


__all__ = [
    "CONTENT",
    "METRICS",
    "REVIEW",
    "OversightAccess",
    "OversightDefaultPermission",
    "reader_id",
]
