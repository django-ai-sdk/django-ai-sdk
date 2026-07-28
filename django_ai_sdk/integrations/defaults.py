"""Integrations for common third-party services, maintained by the SDK.

These exist so no project has to hand-roll its own Notion or Linear MCP setup — the
URL and auth style are the parts everyone would otherwise copy between projects.
Enable one by naming it in ``AI_SDK_INTEGRATIONS`` and putting its credentials in the
matching settings slice::

    AI_SDK_INTEGRATIONS = {"notion": "django_ai_sdk.integrations.defaults.NotionService"}
    AI_SDK_NOTION = {"tools": ["notion-search"]}

Every attribute below is overridable per deployment through that slice (see
``MCPIntegrationService._get_params``), so a project needing a different URL, scope or
tool allow-list does not need its own subclass.

Your own integrations — product-specific APIs, MCP servers only you run — belong in
your own app, not here. Subclass ``MCPIntegrationService`` for an MCP server or
``APIIntegration`` for a hand-written client.
"""

from __future__ import annotations

from django_ai_sdk.integrations.mcp.loader import MCPIntegrationService


class NotionService(MCPIntegrationService):
    """Notion's hosted MCP server. OAuth 2.1 + PKCE, so there is no static secret to
    configure — each user connects their own account via ``/api/integrations``."""

    name = "notion"
    label = "Notion"
    url = "https://mcp.notion.com/mcp"
    auth = "oauth"
    default_tools = ["notion-search"]


class LinearService(MCPIntegrationService):
    """Linear's hosted MCP server, via a shared deployment token.

    ``AI_SDK_LINEAR = {"token": ...}`` — a missing token doesn't crash boot; the
    integration registers and reports itself as needing setup instead.
    """

    name = "linear"
    label = "Linear"
    url = "https://mcp.linear.app/mcp"
    auth = "token"
    default_tools = ["list_issues"]
