from __future__ import annotations

from django_ai_sdk.integrations.apps import IntegrationAppConfig


class GitHubConfig(IntegrationAppConfig):
    default = True
    name = "django_ai_sdk.integrations.github"
    # Prefixed to avoid clashing with host app labels.
    label = "django_ai_sdk_github"
    integration = "django_ai_sdk.integrations.github.integration.GitHubIntegration"
