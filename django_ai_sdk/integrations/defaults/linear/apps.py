from __future__ import annotations

from django_ai_sdk.integrations.apps import IntegrationAppConfig


class LinearConfig(IntegrationAppConfig):
    name = "django_ai_sdk.integrations.defaults.linear"
    label = "django_ai_sdk_linear"
    verbose_name = "Django AI SDK — Linear"
    service = "django_ai_sdk.integrations.defaults.linear.services.LinearService"
