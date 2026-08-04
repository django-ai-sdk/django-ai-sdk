"""GitHub's Copilot MCP server — repos, issues, etc. via a personal access token."""

from __future__ import annotations

from django_ai_sdk.integrations.mcp.loader import MCPIntegration


class GitHubIntegration(MCPIntegration):
    """Defaults to a shared personal access token, read from
    AI_SDK_INTEGRATIONS["github"]["TOKEN"].

    GitHub also supports per-user OAuth against the same server. That is a deployment
    choice, not a different integration, so switch it in settings rather than
    subclassing -- see MCPIntegration on the AUTH key::

        AI_SDK_INTEGRATIONS = {"github": {
            "AUTH": "oauth",
            "CLIENT_ID": env("GITHUB_OAUTH_CLIENT_ID"),
            "CLIENT_SECRET": env("GITHUB_OAUTH_CLIENT_SECRET"),
        }}

    default_tools = [] means no allow-list: every tool the server offers is
    discovered rather than a fixed subset.
    """

    name = "github"
    label = "GitHub"
    url = "https://api.githubcopilot.com/mcp"
    auth = "token"
    default_tools = []
