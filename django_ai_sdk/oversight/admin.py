"""Read-only admin for the oversight record.

It needs only metrics access, so it never shows content (turn text, reasons,
notes) or who asked in which thread.
"""

from __future__ import annotations

from typing import Any

from django.contrib import admin

from django_ai_sdk.oversight.models import AccessRecord, LensAnswer, ReviewItem, RuleMatch, Turn


class ReadOnlyAdmin(admin.ModelAdmin):
    def has_add_permission(self, request: Any, obj: Any = None) -> bool:
        return False

    def has_change_permission(self, request: Any, obj: Any = None) -> bool:
        return False

    def has_delete_permission(self, request: Any, obj: Any = None) -> bool:
        return False


@admin.register(Turn)
class TurnAdmin(ReadOnlyAdmin):
    list_display = ("created_at", "channel", "agent_name", "outcome", "report_outcome", "severity")
    list_filter = ("outcome", "report_outcome", "severity", "channel", "policy")
    # The review API shows these ids only with VIEW_OVERSIGHT_CONTENT.
    exclude = ("content", "actor_id", "thread_id", "user_message_id", "message_id")


@admin.register(LensAnswer)
class LensAnswerAdmin(ReadOnlyAdmin):
    # Reasons live in the turn's content, which the admin does not show.
    list_display = ("created_at", "lens", "stage", "answerer", "value", "confidence", "error")
    list_filter = ("lens", "stage", "answerer")


@admin.register(RuleMatch)
class RuleMatchAdmin(ReadOnlyAdmin):
    list_display = ("created_at", "rule", "stage", "mode", "acted")
    list_filter = ("rule", "mode", "acted")


@admin.register(ReviewItem)
class ReviewItemAdmin(ReadOnlyAdmin):
    # The reviewer's note lives in the turn's content, which the admin does not show.
    list_display = ("created_at", "queue", "lens", "rule", "priority", "status")
    list_filter = ("queue", "status")


@admin.register(AccessRecord)
class AccessRecordAdmin(ReadOnlyAdmin):
    # Reads and changes through the API. Views in this admin are not recorded.
    list_display = ("created_at", "reader_label", "operation", "content")
    list_filter = ("operation", "content")
