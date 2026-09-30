from __future__ import annotations

from typing import TYPE_CHECKING

from django.contrib import admin
from django_ai_sdk.memories.models import Memory, MemoryGroup, MemoryUser

if TYPE_CHECKING:
    from django.db.models import QuerySet
    from django.http import HttpRequest


class MemoryUserInline(admin.TabularInline):
    model = MemoryUser
    extra = 0
    autocomplete_fields = ["user"]
    verbose_name = "Knowledge user"
    verbose_name_plural = "Knowledge users"


class MemoryGroupInline(admin.TabularInline):
    model = MemoryGroup
    extra = 0
    autocomplete_fields = ["group"]
    verbose_name = "Knowledge group"
    verbose_name_plural = "Knowledge groups"


@admin.register(Memory)
class KnowledgeAdmin(admin.ModelAdmin):
    list_display = ["name", "slug", "is_public", "created_at"]
    search_fields = ["name", "slug"]
    readonly_fields = ["slug", "created_at", "updated_at"]
    exclude = ["is_hidden"]
    inlines = [MemoryUserInline, MemoryGroupInline]

    def get_queryset(self, request: HttpRequest) -> QuerySet:
        return super().get_queryset(request).filter(is_hidden=False)
