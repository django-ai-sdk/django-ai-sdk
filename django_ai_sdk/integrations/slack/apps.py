from __future__ import annotations

from django_ai_sdk.integrations.apps import IntegrationAppConfig


class SlackConfig(IntegrationAppConfig):
    default = True
    name = "django_ai_sdk.integrations.slack"
    # Prefixed so the host project can still use the label "slack" itself.
    label = "django_ai_sdk_slack"
    integration = "django_ai_sdk.integrations.slack.integration.SlackIntegration"
