from __future__ import annotations

from django_ai_sdk.integrations.apps import IntegrationAppConfig


class NotionConfig(IntegrationAppConfig):
    default = True
    name = "django_ai_sdk.integrations.notion"
    integration = "django_ai_sdk.integrations.notion.integration.NotionIntegration"
