"""The oversight app. On start it loads the packs, each app's `oversight.py`, the
system checks and the deletion signals."""

from __future__ import annotations

from django.apps import AppConfig
from django.core.checks import register as register_check
from django.utils.module_loading import autodiscover_modules


class OversightConfig(AppConfig):
    """Opt-in app that checks and records agent turns."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "django_ai_sdk.oversight"
    label = "django_ai_sdk_oversight"
    verbose_name = "Oversight"

    def ready(self) -> None:
        # Load the packs first so an app's policy can extend them by name.
        import django_ai_sdk.oversight.packs  # noqa: F401

        autodiscover_modules("oversight")

        from django_ai_sdk.oversight import signals
        from django_ai_sdk.oversight.checks import check_oversight, check_oversight_deploy

        register_check(check_oversight)
        register_check(check_oversight_deploy, deploy=True)
        # Deleting a thread forgets what oversight holds about it.
        signals.connect()
