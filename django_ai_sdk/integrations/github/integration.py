from __future__ import annotations

from django_ai_sdk.integrations.mcp.loader import MCPIntegration


class GitHubIntegration(MCPIntegration):
    """GitHub MCP server; shared TOKEN by default, or "AUTH": "oauth" (see MCPIntegration)."""

    name = "github"
    label = "GitHub"
    url = "https://api.githubcopilot.com/mcp"
    auth = "token"
    default_tools = []
