from __future__ import annotations

from django.contrib import admin

from django_ai_sdk.conversation.models import Message, MessageFeedback, Thread


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
