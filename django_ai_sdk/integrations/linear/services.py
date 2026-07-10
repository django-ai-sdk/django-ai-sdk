"""Linear integration — a static-token MCP server.

Params come from the ``AI_SDK_LINEAR`` settings slice, e.g.::

    AI_SDK_LINEAR = {"token": config.getliteral("app", "linear_api_key")}
"""

from __future__ import annotations

from django_ai_sdk.integrations.mcp.loader import MCPIntegrationService


class LinearService(MCPIntegrationService):
    name = "linear"
    label = "Linear"
    url = "https://mcp.linear.app/mcp"
    auth = "token"
    default_tools: list[str] = []
