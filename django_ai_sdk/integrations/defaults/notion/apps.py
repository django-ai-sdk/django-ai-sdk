from __future__ import annotations

from django_ai_sdk.integrations.apps import IntegrationAppConfig


class NotionConfig(IntegrationAppConfig):
    name = "django_ai_sdk.integrations.defaults.notion"
    label = "django_ai_sdk_notion"
    verbose_name = "Django AI SDK — Notion"
    service = "django_ai_sdk.integrations.defaults.notion.services.NotionService"
