"""Read-only admin that lists every authored policy version.

Saves go through `AuthoringService`, which validates the definition, requires a reason
and writes the access record. The admin never changes a row.
"""

from __future__ import annotations

from django.contrib import admin

from django_ai_sdk.oversight.admin import ReadOnlyAdmin
from django_ai_sdk.oversight.authoring.models import AuthoredPolicy


@admin.register(AuthoredPolicy)
class AuthoredPolicyAdmin(ReadOnlyAdmin):
    list_display = ("name", "version", "label", "withdrawn", "changed_at", "changed_by_id")
    list_filter = ("withdrawn",)
    search_fields = ("name", "label", "reason")
    date_hierarchy = "changed_at"


__all__ = ["AuthoredPolicyAdmin"]
