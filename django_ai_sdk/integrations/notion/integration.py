"""Notion's hosted MCP server."""

from __future__ import annotations

from django_ai_sdk.integrations.mcp.loader import MCPIntegration


class NotionIntegration(MCPIntegration):
    """OAuth 2.1 + PKCE, so there is no static secret to configure. Each user
    connects their own account via /api/integrations."""

    name = "notion"
    label = "Notion"
    url = "https://mcp.notion.com/mcp"
    auth = "oauth"
    default_tools = ["notion-search"]
