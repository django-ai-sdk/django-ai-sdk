from __future__ import annotations

from django_ai_sdk.integrations.apps import IntegrationAppConfig


class TelegramConfig(IntegrationAppConfig):
    default = True
    name = "django_ai_sdk.integrations.telegram"
    # Prefixed so the host project can still use the label "telegram" itself.
    label = "django_ai_sdk_telegram"
    integration = "django_ai_sdk.integrations.telegram.integration.TelegramIntegration"
