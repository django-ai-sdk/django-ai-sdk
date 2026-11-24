from __future__ import annotations

from django_ai_sdk.integrations.apps import IntegrationAppConfig


class LinearConfig(IntegrationAppConfig):
    default = True
    name = "django_ai_sdk.integrations.linear"
    # Prefixed to avoid clashing with host app labels.
    label = "django_ai_sdk_linear"
    integration = "django_ai_sdk.integrations.linear.integration.LinearIntegration"
