from __future__ import annotations

from django_ai_sdk.integrations.apps import IntegrationAppConfig


class GitHubConfig(IntegrationAppConfig):
    default = True
    name = "django_ai_sdk.integrations.github"
    integration = "django_ai_sdk.integrations.github.integration.GitHubIntegration"
