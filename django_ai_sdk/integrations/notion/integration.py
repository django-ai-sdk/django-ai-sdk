from __future__ import annotations

from django_ai_sdk.integrations.mcp.loader import MCPIntegration


class NotionIntegration(MCPIntegration):
    name = "notion"
    label = "Notion"
    url = "https://mcp.notion.com/mcp"
    auth = "oauth"
    default_tools = ["notion-search"]
