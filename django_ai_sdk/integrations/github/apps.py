from __future__ import annotations

from django_ai_sdk.integrations.apps import IntegrationAppConfig


class GitHubConfig(IntegrationAppConfig):
    default = True
    name = "django_ai_sdk.integrations.github"
    # Namespaced so the SDK never claims the global app label "github", which a
    # host project may well want for an app of its own (as mcp/apps.py does).
    label = "django_ai_sdk_github"
    integration = "django_ai_sdk.integrations.github.integration.GitHubIntegration"
