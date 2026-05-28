"""MCP backend interface and implementations.

Custom backends should implement the MCPBackend protocol:

    from typing import Any
    from django_ai_sdk.mcp.backends import MCPBackend

    class CustomBackend(MCPBackend):
        async def connect(
            self, url: str, token: str | None, tools: list[str] | None
        ) -> list[Any]:
            pass

Then point AI_SDK_MCP_BACKEND at it in settings.py:

    AI_SDK_MCP_BACKEND = 'path.to.CustomBackend'
"""

from __future__ import annotations

from typing import Any, Protocol


class MCPBackend(Protocol):
    """Protocol for MCP backend implementations."""

    async def connect(self, url: str, token: str | None, tools: list[str] | None) -> list[Any]:
        """Connect to an MCP server and return tool objects.

        Args:
            url: The MCP server URL.
            token: Optional bearer token for authentication.
            tools: Optional list of tool names to load. If None, load all.

        Returns:
            A list of tool objects compatible with the framework.
        """
        pass
