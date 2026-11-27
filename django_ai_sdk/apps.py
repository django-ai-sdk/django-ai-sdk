from __future__ import annotations

from django.apps import AppConfig
from django.core.checks import register as register_check
from django.utils.module_loading import autodiscover_modules


class DjangoAISDKConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "django_ai_sdk"

    def ready(self) -> None:
        import django_ai_sdk.storage.db  # noqa: F401
        import django_ai_sdk.storage.memory  # noqa: F401

        autodiscover_modules("agents")

        from django_ai_sdk.agents.registry import registry

        registry.setup()

        from django_ai_sdk.checks import check_oversight_installed, check_webhook_dedup_cache

        register_check(check_oversight_installed)
        register_check(check_webhook_dedup_cache)
