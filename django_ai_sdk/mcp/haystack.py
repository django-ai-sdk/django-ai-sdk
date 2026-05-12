from haystack.utils import Secret
from haystack_integrations.tools.mcp import MCPToolset, StreamableHttpServerInfo


def get_mcp_server(url: str, tools: list[str], token: str | None = None) -> MCPToolset:
    """
    Creates an MCPToolset for the given server URL.

    Args:
        url (str): The URL of the MCP server.

    Returns:
        MCPToolset: An instance of MCPToolset configured with the given server URL.
    """

    server_info = StreamableHttpServerInfo(
        url=url,
        token=Secret.from_token(token) if token else None,
    )

    # TODO: check if we need to warmup? This would be better then eager_connect
    return MCPToolset(server_info=server_info, tool_names=tools, eager_connect=True)
