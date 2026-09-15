from __future__ import annotations

from django_ai_sdk.integrations.apps import IntegrationAppConfig


class SlackConfig(IntegrationAppConfig):
    default = True
    name = "django_ai_sdk.integrations.slack"
    # Namespaced so the SDK never claims the global app label "slack", which a
    # host project may well want for an app of its own (as mcp/apps.py does).
    label = "django_ai_sdk_slack"
    integration = "django_ai_sdk.integrations.slack.integration.SlackIntegration"
