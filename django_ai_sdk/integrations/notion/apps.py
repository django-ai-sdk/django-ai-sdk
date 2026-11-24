from __future__ import annotations

from django_ai_sdk.integrations.apps import IntegrationAppConfig


class NotionConfig(IntegrationAppConfig):
    default = True
    name = "django_ai_sdk.integrations.notion"
    # Prefixed to avoid clashing with host app labels.
    label = "django_ai_sdk_notion"
    integration = "django_ai_sdk.integrations.notion.integration.NotionIntegration"
