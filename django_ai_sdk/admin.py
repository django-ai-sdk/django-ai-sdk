from django.contrib import admin

from django_ai_sdk.conversation.models import Message, Thread


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ("id", "thread", "get_role", "rating", "rating_comment_preview", "created_at")
    list_filter = ("rating",)
    search_fields = ("rating_comment",)
    readonly_fields = ("id", "thread", "result", "created_at")
    fields = ("id", "thread", "result", "rating", "rating_comment", "is_deleted", "created_at")

    def get_role(self, obj: Message) -> str:
        return obj.result.get("role", "")

    get_role.short_description = "Role"

    def rating_comment_preview(self, obj: Message) -> str:
        if len(obj.rating_comment) > 60:
            return obj.rating_comment[:60] + "..."
        return obj.rating_comment

    rating_comment_preview.short_description = "Comment"


@admin.register(Thread)
class ThreadAdmin(admin.ModelAdmin):
    list_display = ("id", "title", "user", "message_count", "created_at")
    search_fields = ("title",)
    readonly_fields = ("id", "created_at", "updated_at")
