from __future__ import annotations

from django_ai_sdk.integrations.apps import IntegrationAppConfig


class LinearConfig(IntegrationAppConfig):
    default = True
    name = "django_ai_sdk.integrations.linear"
    integration = "django_ai_sdk.integrations.linear.integration.LinearIntegration"
