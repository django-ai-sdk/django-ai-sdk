from __future__ import annotations

from django.apps import AppConfig


class DjangoAISDKConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "django_ai_sdk"

    def ready(self) -> None:
        # Register signal handlers (e.g. MessageImage file cleanup on delete).
        from django_ai_sdk.conversation import signals  # noqa: F401
