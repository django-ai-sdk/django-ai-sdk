"""Who may read the oversight record, read its content, and answer its review queues.

The default for the `oversight` permission domain: each operation needs the Django
model permission in `MODEL_PERMISSIONS`, never `is_staff`. Subclass it and override
`scope_turns`/`scope_reviews` to narrow which turns a reader reaches (the oversight
manual's Review API).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django_ai_sdk.permissions import BasePermission, Operation, ahas_model_perm

if TYPE_CHECKING:
    from django.db.models import QuerySet

    from django_ai_sdk.types import UserType


class OversightDefaultPermission(BasePermission):
    """Check a model permission per operation, and narrow turns with the scope hooks."""

    MODEL_PERMISSIONS: dict[Operation, str] = {  # noqa: RUF012
        Operation.VIEW_OVERSIGHT_METRICS: "django_ai_sdk_oversight.view_turn",
        Operation.VIEW_OVERSIGHT_CONTENT: "django_ai_sdk_oversight.view_turncontent",
        Operation.REVIEW_OVERSIGHT: "django_ai_sdk_oversight.change_reviewitem",
    }

    async def has_permission(self, user: UserType, operation: Operation, **kwargs: Any) -> bool:
        perm = self.MODEL_PERMISSIONS.get(operation)
        return perm is not None and await ahas_model_perm(user, perm)

    async def has_object_permission(
        self, user: UserType, operation: Operation, obj: Any, **kwargs: Any
    ) -> bool:
        """Allow a turn or review item when the scope hooks keep it."""
        from asgiref.sync import sync_to_async

        from django_ai_sdk.oversight.models import ReviewItem, Turn

        if not isinstance(obj, Turn | ReviewItem):
            return False
        scoped = self.get_queryset_perms(user, operation, type(obj).objects.filter(pk=obj.pk))
        # Run in a thread, since a hook may run its own queries.
        return await sync_to_async(scoped.exists)()

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


__all__ = ["OversightDefaultPermission"]
