"""Read-only admin for the oversight record.

It needs only metrics access, so it never shows content (turn text, reasons,
notes) or who asked in which thread. Turns and their rows are narrowed by the
same scope hooks as the review API.
"""

from __future__ import annotations

from typing import Any

from django.contrib import admin

from django_ai_sdk.oversight.models import LensAnswer, ReviewItem, RuleMatch, Turn


class ReadOnlyAdmin(admin.ModelAdmin):
    def has_add_permission(self, request: Any, obj: Any = None) -> bool:
        return False

    def has_change_permission(self, request: Any, obj: Any = None) -> bool:
        return False

    def has_delete_permission(self, request: Any, obj: Any = None) -> bool:
        return False


class ScopedAdmin(ReadOnlyAdmin):
    """Rows of the turns the reader's METRICS scope keeps, as in the review API."""

    # The path from a row to its turn; "" for the turn itself.
    turn_path = "turn"

    def get_queryset(self, request: Any) -> Any:
        from django_ai_sdk.oversight.services import OversightService
        from django_ai_sdk.oversight.services.access import METRICS

        turns = OversightService.has_queryset_perms(
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
    # Reasons live in the turn's content, which the admin does not show.
    list_display = ("created_at", "lens", "stage", "answerer", "value", "confidence", "error")
    list_filter = ("lens", "stage", "answerer")


@admin.register(RuleMatch)
class RuleMatchAdmin(ScopedAdmin):
    list_display = ("created_at", "rule", "stage", "mode", "acted")
    list_filter = ("rule", "mode", "acted")


@admin.register(ReviewItem)
class ReviewItemAdmin(ScopedAdmin):
    # The reviewer's note is content, which the admin does not show.
    list_display = ("created_at", "queue", "rule", "status", "agree")
    list_filter = ("queue", "status")
    # The note is content, and who closed it is kept out like who asked.
    exclude = ("note", "resolved_by_id")
