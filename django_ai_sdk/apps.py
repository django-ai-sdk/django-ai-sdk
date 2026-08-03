from __future__ import annotations

from django.apps import AppConfig
from django.utils.module_loading import autodiscover_modules


class DjangoAISDKConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "django_ai_sdk"

    def ready(self) -> None:
        autodiscover_modules("assistants")

        from django_ai_sdk.assistants.registry import registry

        registry.setup()
