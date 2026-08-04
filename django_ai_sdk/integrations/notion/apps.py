from __future__ import annotations

from django_ai_sdk.integrations.apps import IntegrationAppConfig


class NotionConfig(IntegrationAppConfig):
    default = True
    name = "django_ai_sdk.integrations.notion"
    # Namespaced so the SDK never claims the global app label "notion", which a
    # host project may well want for an app of its own (as mcp/apps.py does).
    label = "django_ai_sdk_notion"
    integration = "django_ai_sdk.integrations.notion.integration.NotionIntegration"
