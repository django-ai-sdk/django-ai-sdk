from __future__ import annotations

from typing import TYPE_CHECKING

from django.contrib import admin

from django_ai_sdk.conversation.models import Message, MessageFeedback, Thread
from django_ai_sdk.workflows.models import WorkflowRun, WorkflowRunStep, WorkflowSettings

if TYPE_CHECKING:
    from django.http import HttpRequest


class MessageFeedbackInline(admin.TabularInline):
    model = MessageFeedback
    extra = 1
    readonly_fields = ("id", "created_at")
    fields = ("user", "rating", "feedback", "created_at")


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ("id", "thread", "get_role", "created_at")
    search_fields = ("result",)
    readonly_fields = ("id", "thread", "result", "created_at")
    fields = ("id", "thread", "result", "is_deleted", "created_at")
    inlines = (MessageFeedbackInline,)

    def get_role(self, obj: Message) -> str:
        return obj.result.get("role", "")


@admin.register(MessageFeedback)
class MessageFeedbackAdmin(admin.ModelAdmin):
    list_display = ("id", "message", "user", "rating", "feedback_preview", "created_at")
    list_filter = ("rating", "created_at")
    search_fields = ("feedback",)
    readonly_fields = ("id", "message", "created_at")
    fields = ("id", "message", "user", "rating", "feedback", "created_at")

    @admin.display(description="Feedback")
    def feedback_preview(self, obj: MessageFeedback) -> str:
        if len(obj.feedback) > 60:
            return obj.feedback[:60] + "..."
        return obj.feedback


@admin.register(Thread)
class ThreadAdmin(admin.ModelAdmin):
    list_display = ("id", "title", "user", "message_count", "created_at")
    search_fields = ("title",)
    readonly_fields = ("id", "created_at", "updated_at")


class WorkflowRunStepInline(admin.TabularInline):
    """Per-step detail, read-only: a run is a record of what happened."""

    model = WorkflowRunStep
    extra = 0
    can_delete = False
    fields = ("sequence", "step_name", "output_key", "status", "error", "completed_at")
    readonly_fields = fields
    ordering = ("sequence",)

    def has_add_permission(self, request: HttpRequest, obj: object = None) -> bool:
        return False


@admin.register(WorkflowSettings)
class WorkflowSettingsAdmin(admin.ModelAdmin):
    """Workflows stored in the database.

    `slug` is read-only: changing the registry key repoints every reference to it.
    """

    list_display = ("name", "slug", "active", "created_by", "updated_at")
    list_filter = ("active",)
    search_fields = ("name", "slug")
    readonly_fields = ("id", "slug", "created_at", "updated_at")
    fields = (
        "id",
        "name",
        "slug",
        "active",
        "definition",
        "created_by",
        "created_at",
        "updated_at",
    )


@admin.register(WorkflowRun)
class WorkflowRunAdmin(admin.ModelAdmin):
    list_display = ("id", "workflow", "status", "user", "created_at", "completed_at")
    list_filter = ("status",)
    search_fields = ("error",)
    date_hierarchy = "created_at"
    readonly_fields = tuple(f.name for f in WorkflowRun._meta.fields)
    inlines = (WorkflowRunStepInline,)

    def has_add_permission(self, request: HttpRequest) -> bool:
        return False

    def has_change_permission(self, request: HttpRequest, obj: object = None) -> bool:
        return False
