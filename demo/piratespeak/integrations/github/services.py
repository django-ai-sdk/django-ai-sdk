"""APIIntegration example whose tool is a per-request factory, not a static one."""

from __future__ import annotations

from django_ai_sdk.integrations.api.base import APIIntegration

from piratespeak.integrations.github.api import check_github_api
from piratespeak.integrations.github.tools import list_my_repos


class GithubService(APIIntegration):
    """APIIntegration example demonstrating the factory-tool shape (see tools.py)."""

    name = "github"
    label = "GitHub"
    tools = [list_my_repos]
    health_check = staticmethod(check_github_api)
