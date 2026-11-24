"""Read-only admin for the oversight record: metrics access only, so no content or who asked."""

from __future__ import annotations

from typing import Any

from asgiref.sync import async_to_sync
from django.contrib import admin
from django.db.models.constants import LOOKUP_SEP

from django_ai_sdk.oversight.models import LensAnswer, ReviewItem, RuleMatch, Turn
from django_ai_sdk.oversight.permissions import METRICS, OversightAccess


class ReadOnlyAdmin(admin.ModelAdmin):
    def has_add_permission(self, request: Any, obj: Any = None) -> bool:
        return False

    def has_change_permission(self, request: Any, obj: Any = None) -> bool:
        return False

    def has_delete_permission(self, request: Any, obj: Any = None) -> bool:
        return False


class ScopedAdmin(ReadOnlyAdmin):
    """Rows of the turns the reader's METRICS scope keeps, on top of Django's view permission."""

    # The path from a row to its turn; "" for the turn itself.
    turn_path = "turn"

    def may_read(self, request: Any) -> bool:
        return async_to_sync(OversightAccess.has_perms)(request.user, METRICS, raise_on_deny=False)

    def has_module_permission(self, request: Any) -> bool:
        return super().has_module_permission(request) and self.may_read(request)

    def has_view_permission(self, request: Any, obj: Any = None) -> bool:
        return super().has_view_permission(request, obj) and self.may_read(request)

    def lookup_allowed(self, lookup: str, value: str, request: Any) -> bool:
        # Django allows any field in the querystring, so ?note__icontains= would search a hidden one.
        field = lookup.split(LOOKUP_SEP, 1)[0]
        return field in self.list_filter and super().lookup_allowed(lookup, value, request)

    def get_queryset(self, request: Any) -> Any:
        turns = OversightAccess.has_queryset_perms(
            request.user, METRICS, queryset=Turn.objects.all()
        )
        lookup = f"{self.turn_path}__in" if self.turn_path else "pk__in"
        return super().get_queryset(request).filter(**{lookup: turns.values("pk")})


@admin.register(Turn)
class TurnAdmin(ScopedAdmin):
    turn_path = ""
    list_display = ("created_at", "channel", "agent_name", "outcome", "report_outcome", "severity")
    list_filter = ("outcome", "report_outcome", "severity", "channel", "policy")
    # The review API shows these ids only with VIEW_OVERSIGHT_CONTENT.
    exclude = ("content", "actor_id", "thread_id", "user_message_id", "message_id")


@admin.register(LensAnswer)
class LensAnswerAdmin(ScopedAdmin):
    list_display = ("created_at", "lens", "stage", "answerer", "value", "confidence", "error")
    list_filter = ("lens", "stage", "answerer")


@admin.register(RuleMatch)
class RuleMatchAdmin(ScopedAdmin):
    list_display = ("created_at", "rule", "stage", "mode", "acted")
    list_filter = ("rule", "mode", "acted")


@admin.register(ReviewItem)
class ReviewItemAdmin(ScopedAdmin):
    list_display = ("created_at", "queue", "rule", "status", "agree")
    list_filter = ("queue", "status")
    # The note is content, and who closed it is kept out like who asked.
    exclude = ("note", "resolved_by_id")
