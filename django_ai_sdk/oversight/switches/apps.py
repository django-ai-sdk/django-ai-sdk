"""Switches app config. Adds its permissions and routes to oversight, and installs the
database overlay and the system check."""

from __future__ import annotations

from django.apps import AppConfig
from django.core.checks import register as register_check


class SwitchesConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "django_ai_sdk.oversight.switches"
    label = "django_ai_sdk_oversight_switches"
    verbose_name = "Oversight switches"

    def ready(self) -> None:
        from django_ai_sdk.oversight.apps import extend, require_apps
        from django_ai_sdk.permissions import Operation

        require_apps(self.name, "django_ai_sdk.oversight")
        extend(
            permissions={
                Operation.VIEW_OVERSIGHT_SWITCHES: f"{self.label}.view_switch",
                Operation.SWITCH_OVERSIGHT: f"{self.label}.change_switch",
            },
            routes="django_ai_sdk.contrib.ninja.oversight_switches",
        )
        from django_ai_sdk.oversight.resolve import set_overlay_source
        from django_ai_sdk.oversight.switches.checks import check_switches
        from django_ai_sdk.oversight.switches.overlay import DatabaseOverlay

        register_check(check_switches)
        set_overlay_source(DatabaseOverlay())
