from __future__ import annotations

from django_ai_sdk.integrations.apps import IntegrationAppConfig


class GitHubEventsConfig(IntegrationAppConfig):
    default = True
    name = "django_ai_sdk.integrations.github_events"
    # Prefixed to avoid host labels; apart from the `github` MCP app so each installs alone.
    label = "django_ai_sdk_github_events"
    integration = "django_ai_sdk.integrations.github_events.integration.GitHubEventsIntegration"
