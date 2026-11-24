from __future__ import annotations

from django_ai_sdk.integrations.mcp.loader import MCPIntegration


class LinearIntegration(MCPIntegration):
    """Linear MCP server with a shared token from AI_SDK_INTEGRATIONS["linear"]["TOKEN"]."""

    name = "linear"
    label = "Linear"
    url = "https://mcp.linear.app/mcp"
    auth = "token"
    default_tools = ["list_issues"]
