from __future__ import annotations

from django.contrib import admin

from django_ai_sdk.oversight.admin import ReadOnlyAdmin
from django_ai_sdk.oversight.authoring.models import AuthoredPolicy


@admin.register(AuthoredPolicy)
class AuthoredPolicyAdmin(ReadOnlyAdmin):
    list_display = ("name", "label", "active", "updated_at", "updated_by_id")
    list_filter = ("active",)
    search_fields = ("name", "label")
    date_hierarchy = "updated_at"
