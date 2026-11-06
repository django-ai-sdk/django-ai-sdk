"""Read-only admin that lists every switch version.

Switches go through `SwitchService`, which enforces the floor, requires a reason and
writes the access record. The admin never changes a row.
"""

from __future__ import annotations

from django.contrib import admin

from django_ai_sdk.oversight.admin import ReadOnlyAdmin
from django_ai_sdk.oversight.switches.models import Override


@admin.register(Override)
class OverrideAdmin(ReadOnlyAdmin):
    list_display = ("changed_at", "kind", "policy", "target", "version", "value", "changed_by_id")
    list_filter = ("kind", "policy")
    search_fields = ("target", "policy", "reason")
    date_hierarchy = "changed_at"


__all__ = ["OverrideAdmin"]
