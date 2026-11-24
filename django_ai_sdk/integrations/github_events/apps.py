from __future__ import annotations

from django_ai_sdk.integrations.apps import IntegrationAppConfig


class GitHubEventsConfig(IntegrationAppConfig):
    default = True
    name = "django_ai_sdk.integrations.github_events"
    # Prefixed to avoid clashing with host app labels.
    label = "django_ai_sdk_github_events"
    integration = "django_ai_sdk.integrations.github_events.integration.GitHubEventsIntegration"
