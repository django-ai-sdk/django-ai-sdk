from __future__ import annotations

from django_ai_sdk.integrations.apps import IntegrationAppConfig


class GitHubEventsConfig(IntegrationAppConfig):
    default = True
    name = "django_ai_sdk.integrations.github_events"
    label = "django_ai_sdk_github_events"
    integration = "django_ai_sdk.integrations.github_events.integration.GitHubEventsIntegration"
