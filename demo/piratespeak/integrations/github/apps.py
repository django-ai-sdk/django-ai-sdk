from __future__ import annotations

from django_ai_sdk.integrations.apps import IntegrationAppConfig


class GithubConfig(IntegrationAppConfig):
    name = "piratespeak.integrations.github"
    label = "piratespeak_github"
    verbose_name = "Piratespeak — GitHub"
    service = "piratespeak.integrations.github.services.GithubService"
