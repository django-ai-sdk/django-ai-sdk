"""GitHub's Copilot MCP server — repos, issues, etc. via a personal access token."""

from __future__ import annotations

from django_ai_sdk.integrations.mcp.loader import MCPIntegration


class GitHubIntegration(MCPIntegration):
    """auth = "token" reads the PAT via get_integration_secret("github", "token").

    default_tools = [] means no allow-list: every tool the server offers is
    discovered rather than a fixed subset.
    """

    name = "github"
    label = "GitHub"
    url = "https://api.githubcopilot.com/mcp"
    auth = "token"
    default_tools = []
