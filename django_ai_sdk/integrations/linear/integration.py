"""Linear's hosted MCP server, via a shared deployment token."""

from __future__ import annotations

from django_ai_sdk.integrations.mcp.loader import MCPIntegration


class LinearIntegration(MCPIntegration):
    """Reads AI_SDK_INTEGRATIONS["linear"]["TOKEN"]. A missing token doesn't crash
    boot; the integration registers and reports itself as needing setup instead."""

    name = "linear"
    label = "Linear"
    url = "https://mcp.linear.app/mcp"
    auth = "token"
    default_tools = ["list_issues"]
