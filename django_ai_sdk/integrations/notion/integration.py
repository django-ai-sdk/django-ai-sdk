"""Notion's hosted MCP server."""

from __future__ import annotations

from django_ai_sdk.integrations.mcp.loader import MCPIntegration


class NotionIntegration(MCPIntegration):
    """OAuth 2.1 + PKCE: each user connects their own account, so there is no shared
    token to configure. Notion supports dynamic client registration (RFC 7591), so
    AI_SDK_INTEGRATIONS["notion"] can stay empty; set CLIENT_ID and CLIENT_SECRET
    there only to use a pre-registered app instead."""

    name = "notion"
    label = "Notion"
    url = "https://mcp.notion.com/mcp"
    auth = "oauth"
    default_tools = ["notion-search"]
