"""Switches app config. Installs the database overlay and the system check."""

from __future__ import annotations

from django.apps import AppConfig
from django.core.checks import register as register_check


class SwitchesConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "django_ai_sdk.oversight.switches"
    label = "django_ai_sdk_oversight_switches"
    verbose_name = "Oversight switches"

    def ready(self) -> None:
        from django_ai_sdk.oversight.apps import require_apps

        require_apps(self.name, "django_ai_sdk.oversight")
        from django_ai_sdk.oversight.resolve import set_overlay_source
        from django_ai_sdk.oversight.switches.checks import check_switches
        from django_ai_sdk.oversight.switches.overlay import DatabaseOverlay

        register_check(check_switches)
        set_overlay_source(DatabaseOverlay())
