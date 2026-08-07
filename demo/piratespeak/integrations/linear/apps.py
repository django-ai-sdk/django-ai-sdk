from __future__ import annotations

from django_ai_sdk.integrations.apps import IntegrationAppConfig


class LinearConfig(IntegrationAppConfig):
    default = True
    name = "piratespeak.integrations.linear"
    label = "piratespeak_linear"
    integration = "piratespeak.integrations.linear.integration.LinearIntegration"
