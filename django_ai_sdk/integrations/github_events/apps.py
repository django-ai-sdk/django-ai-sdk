from __future__ import annotations

from django_ai_sdk.integrations.apps import IntegrationAppConfig


class GitHubEventsConfig(IntegrationAppConfig):
    default = True
    name = "django_ai_sdk.integrations.github_events"
    # Prefixed so it never takes a label the host project might use, and kept
    # apart from the `github` MCP app so each can be installed on its own.
    label = "django_ai_sdk_github_events"
    integration = "django_ai_sdk.integrations.github_events.integration.GitHubEventsIntegration"
