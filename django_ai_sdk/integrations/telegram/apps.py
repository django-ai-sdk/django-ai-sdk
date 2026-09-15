from __future__ import annotations

from django_ai_sdk.integrations.apps import IntegrationAppConfig


class TelegramConfig(IntegrationAppConfig):
    default = True
    name = "django_ai_sdk.integrations.telegram"
    # Namespaced so the SDK never claims the global app label "telegram", which a
    # host project may well want for an app of its own (as mcp/apps.py does).
    label = "django_ai_sdk_telegram"
    integration = "django_ai_sdk.integrations.telegram.integration.TelegramIntegration"
