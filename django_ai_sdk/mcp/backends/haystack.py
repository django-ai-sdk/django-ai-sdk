"""Default MCP backend — Haystack MCPToolset over Streamable HTTP.

To write a custom backend, create a module that exposes:

    async def connect(url: str, token: str | None, tools: list[str] | None) -> list[Any]: ...

Then point AI_SDK_MCP_BACKEND at it in settings.py.
"""

from __future__ import annotations

import asyncio
from typing import Any


async def connect(url: str, token: str | None, tools: list[str] | None) -> list[Any]:
    """Connect to an MCP server and return tool objects."""
    try:
        from haystack.utils import Secret
        from haystack_integrations.tools.mcp import MCPToolset, StreamableHttpServerInfo
    except ImportError:
        raise ImportError(
            "The default MCP backend requires the haystack extra: "
            "pip install 'django-ai-sdk[mcp,haystack]'"
        ) from None

    def _get_toolset() -> MCPToolset:
        server_info = StreamableHttpServerInfo(
            url=url,
            token=Secret.from_token(token) if token else None,
        )
        return MCPToolset(server_info=server_info, tool_names=tools, eager_connect=True)

    toolset = await asyncio.to_thread(_get_toolset)
    return list(toolset)
