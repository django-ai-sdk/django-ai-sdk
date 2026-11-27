from __future__ import annotations

from django_ai_sdk.integrations.apps import IntegrationAppConfig


class TelegramConfig(IntegrationAppConfig):
    default = True
    name = "django_ai_sdk.integrations.telegram"
    label = "django_ai_sdk_telegram"
    integration = "django_ai_sdk.integrations.telegram.integration.TelegramIntegration"
