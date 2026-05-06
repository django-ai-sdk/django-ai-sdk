import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from django_ai_sdk.logger import get_logger

logger = get_logger(__name__)

__all__ = ["MCPServerConfig", "MCPToolDescriptor", "discover"]


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server.

    Attributes:
        url: The MCP server endpoint URL (Streamable HTTP transport).
        name: Short identifier for this server (e.g. "linear"). Used to prefix
            discovered tool names (``linear__search_issues``) and for per-assistant
            opt-in via ``Assistant.mcp_servers``. Required when running multiple
            servers to prevent tool name collisions.
        bearer_token: Bearer token for Authorization header. Merged with ``headers``;
            takes precedence if both supply an Authorization key.
        headers: Additional HTTP headers sent with every request to this server.
        cache_ttl: Seconds to cache the discovered tool list for this server.
            ``None`` (default) falls back to the ``AI_SDK_MCP_CACHE_TTL`` setting.
            Set to ``0`` to disable caching for this server.
        tools: Explicit allowlist of tool names to expose from this server.
            ``None`` (default) exposes all tools. Use this to reduce the LLM's
            tool surface to only the tools your assistants actually need.
    """

    url: str
    name: str = ""
    bearer_token: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    cache_ttl: int | None = None
    tools: list[str] | None = None

    def get_headers(self) -> dict[str, str]:
        """Return merged HTTP headers, including Authorization if bearer_token is set."""
        h = dict(self.headers)
        if self.bearer_token:
            h["Authorization"] = f"Bearer {self.bearer_token}"
        return h


@dataclass
class MCPToolDescriptor:
    """Framework-agnostic description of a tool discovered from an MCP server.

    Holds the tool metadata and a callable that invokes the tool synchronously.
    Convert to your pipeline framework's tool type before use
    (e.g. ``adapters.haystack.to_haystack_tool``).
    """

    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable[..., str]


# url → (expires_at, list[MCPToolDescriptor])
_cache: dict[str, tuple[float, list[MCPToolDescriptor]]] = {}


def _extract_text(content: list[Any]) -> str:
    """Extract and join text parts from an MCP tool result content list."""
    parts = [item.text for item in content if hasattr(item, "text")]
    return "\n".join(parts) if parts else ""


def _call_tool(
    url: str,
    tool_name: str,
    kwargs: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> str:
    """Invoke an MCP tool synchronously. Safe to call from a thread pool worker.

    Opens a fresh HTTP connection and MCP session for each call (no persistent
    session; see module docstring for the v1 trade-off). Returns the tool output
    as a plain string in all cases — never raises:

    - On success: joined text content from the tool result.
    - On ``result.isError``: ``"[MCP error] <message>"``
    - On network / protocol failure: ``"[MCP unavailable] <exception>"``
    """

    async def _run() -> str:
        from mcp.client.streamable_http import streamable_http_client
        from mcp.shared._httpx_utils import create_mcp_http_client

        from mcp import ClientSession

        try:
            async with create_mcp_http_client(headers=headers) as http_client:
                async with streamable_http_client(url, http_client=http_client) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool(tool_name, kwargs)
                        if result.isError:
                            return f"[MCP error] {_extract_text(result.content)}"
                        return _extract_text(result.content)
        except Exception as exc:
            cause = exc.exceptions[0] if isinstance(exc, BaseExceptionGroup) else exc
            return f"[MCP unavailable] {cause}"

    return asyncio.run(_run())


async def discover(
    config: MCPServerConfig,
    cache_ttl: int = 300,
) -> list[MCPToolDescriptor]:
    """Discover tools from an MCP server and return framework-agnostic descriptors.

    Results are cached in-process by URL for ``cache_ttl`` seconds (pass ``0``
    to disable). On a cache hit the server is not contacted at all.

    Tool names are prefixed with ``config.name`` (e.g. ``linear__search_issues``)
    when ``config.name`` is set, preventing collisions across servers.

    If ``config.tools`` is set, only tools whose original (unprefixed) name
    appears in the allowlist are returned.

    Args:
        config: Server configuration including URL, auth, and filtering options.
        cache_ttl: Seconds to cache the result. Overrides ``config.cache_ttl``
            only when that is not set; callers should prefer passing the resolved
            TTL from ``get_mcp_tools``.

    Returns:
        List of ``MCPToolDescriptor`` objects ready for conversion to a pipeline
        tool type (e.g. via ``adapters.haystack.to_haystack_tool``).
    """
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared._httpx_utils import create_mcp_http_client

    from mcp import ClientSession

    now = time.monotonic()
    cached = _cache.get(config.url)
    if cached and cached[0] > now:
        logger.debug(f"[MCP] {config.url} — served {len(cached[1])} tool(s) from cache")
        return cached[1]

    headers = config.get_headers()

    async with create_mcp_http_client(headers=headers) as http_client:
        async with streamable_http_client(config.url, http_client=http_client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                response = await session.list_tools()

    available = response.tools
    if config.tools is not None:
        allowed = set(config.tools)
        filtered_out = [t.name for t in available if t.name not in allowed]
        available = [t for t in available if t.name in allowed]
        if filtered_out:
            logger.debug(f"[MCP] {config.url} — allowlist filtered out: {filtered_out}")

    prefix = f"{config.name}__" if config.name else ""

    descriptors = [
        MCPToolDescriptor(
            name=f"{prefix}{mcp_tool.name}",
            description=mcp_tool.description or "",
            parameters=mcp_tool.inputSchema,
            function=lambda url=config.url, name=mcp_tool.name, h=headers, **kwargs: _call_tool(
                url, name, kwargs, h
            ),
        )
        for mcp_tool in available
    ]

    if cache_ttl > 0:
        _cache[config.url] = (now + cache_ttl, descriptors)

    logger.info(
        f"[MCP] {config.url} — discovered {len(descriptors)} tool(s): "
        f"{[d.name for d in descriptors]}"
    )
    return descriptors
