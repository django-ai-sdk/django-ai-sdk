from __future__ import annotations

from django_ai_sdk.integrations.mcp.loader import MCPIntegration


class LinearIntegration(MCPIntegration):
    name = "linear"
    label = "Linear"
    url = "https://mcp.linear.app/mcp"
    auth = "token"
    default_tools = ["list_issues"]
