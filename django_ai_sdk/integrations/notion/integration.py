from __future__ import annotations

from django_ai_sdk.integrations.mcp.loader import MCPIntegration


class NotionIntegration(MCPIntegration):
    """Per-user OAuth; dynamic client registration, so CLIENT_ID/CLIENT_SECRET are optional."""

    name = "notion"
    label = "Notion"
    url = "https://mcp.notion.com/mcp"
    auth = "oauth"
    default_tools = ["notion-search"]
