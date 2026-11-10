"""The automations app. On start it imports each app's `automations.py` and registers its check.

Opt-in, on top of the workflows app. Without it in INSTALLED_APPS there are no
automation tables and nothing from this package is imported.
"""

from __future__ import annotations

from django.apps import AppConfig
from django.core.checks import register as register_check
from django.utils.module_loading import autodiscover_modules


class AutomationsConfig(AppConfig):
    """Opt-in automations: registered workflows run on a cron schedule."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "django_ai_sdk.automations"
    label = "django_ai_sdk_automations"
    verbose_name = "Automations"

    def ready(self) -> None:
        autodiscover_modules("automations")

        from django_ai_sdk.automations.checks import check_automations

        register_check(check_automations)
