from __future__ import annotations

from django_ai_sdk.integrations.apps import IntegrationAppConfig


class LinearConfig(IntegrationAppConfig):
    default = True
    name = "django_ai_sdk.integrations.linear"
    # Namespaced so the SDK never claims the global app label "linear", which a
    # host project may well want for an app of its own (as mcp/apps.py does).
    label = "django_ai_sdk_linear"
    integration = "django_ai_sdk.integrations.linear.integration.LinearIntegration"
