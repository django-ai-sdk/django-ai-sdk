"""Authoring app config. Registers the policy loader and the system checks."""

from __future__ import annotations

from django.apps import AppConfig
from django.core.checks import register as register_check


class AuthoringConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "django_ai_sdk.oversight.authoring"
    label = "django_ai_sdk_oversight_authoring"
    verbose_name = "Oversight authoring"

    def ready(self) -> None:
        from django_ai_sdk.oversight.apps import require_apps

        require_apps(self.name, "django_ai_sdk.oversight")
        from django_ai_sdk.oversight.authoring.checks import (
            check_authoring,
            check_authoring_deploy,
        )
        from django_ai_sdk.oversight.authoring.loader import loader
        from django_ai_sdk.oversight.resolve import add_policy_loader

        register_check(check_authoring)
        register_check(check_authoring_deploy, deploy=True)
        add_policy_loader(loader.load)
