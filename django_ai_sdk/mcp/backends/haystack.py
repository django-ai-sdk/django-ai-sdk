"""Default MCP backend — Haystack MCPToolset over Streamable HTTP.

To write a custom backend, implement the MCPBackend protocol from
django_ai_sdk.mcp.backends and point AI_SDK_MCP_BACKEND at it.
"""

from __future__ import annotations

import asyncio
from typing import Any

from haystack.utils import Secret
from haystack_integrations.tools.mcp import MCPToolset, StreamableHttpServerInfo


async def connect(url: str, token: str | None, tools: list[str] | None) -> list[Any]:
    """Connect to an MCP server and return tool objects."""

    def _get_toolset() -> MCPToolset:
        server_info = StreamableHttpServerInfo(
            url=url,
            token=Secret.from_token(token) if token else None,
        )
        return MCPToolset(server_info=server_info, tool_names=tools, eager_connect=True)

    toolset = await asyncio.to_thread(_get_toolset)
    return list(toolset)
