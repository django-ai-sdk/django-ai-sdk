from __future__ import annotations

from django_ai_sdk.integrations.apps import IntegrationAppConfig


class LinearConfig(IntegrationAppConfig):
    default = True
    name = "studio.integrations.linear"
    label = "studio_linear"
    integration = "studio.integrations.linear.integration.LinearIntegration"
