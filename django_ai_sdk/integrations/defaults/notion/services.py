"""Notion integration — an OAuth MCP server.

Params come from the ``AI_SDK_NOTION`` settings slice, e.g.::

    AI_SDK_NOTION = {"tools": ["notion-search"]}   # OAuth: no static secret needed
"""

from __future__ import annotations

from django_ai_sdk.integrations.mcp.loader import MCPIntegrationService


class NotionService(MCPIntegrationService):
    name = "notion"
    label = "Notion"
    url = "https://mcp.notion.com/mcp"
    auth = "oauth"
    default_tools = ["notion-search"]
