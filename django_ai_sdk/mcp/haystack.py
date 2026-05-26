from typing import Any


def get_mcp_server(
    url: str,
    token: str | None = None,
    tools: list[str] | None = None,
) -> Any:
    """
    Create a connected MCPToolset for the given server.

    Uses eager_connect=True (blocking handshake on init). Always call this via
    asyncio.to_thread() from async contexts — Haystack blocks on threading
    primitives during connection setup.
    """
    try:
        from haystack.utils import Secret
        from haystack_integrations.tools.mcp import MCPToolset, StreamableHttpServerInfo
    except ImportError as exc:
        raise ImportError(
            "Haystack is required to connect to MCP servers. "
            "Install it with: pip install django-ai-sdk[haystack]"
        ) from exc

    server_info = StreamableHttpServerInfo(
        url=url,
        token=Secret.from_token(token) if token else None,
    )
    return MCPToolset(server_info=server_info, tool_names=tools, eager_connect=True)
