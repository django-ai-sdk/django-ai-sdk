"""RFC 9728 — OAuth 2.0 Protected Resource Metadata discovery for MCP servers."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

_CACHE_TTL = 3600  # seconds; discovery metadata is stable but can rotate
_cache: dict[str, tuple[float, OAuthDiscovery]] = {}


@dataclass
class OAuthDiscovery:
    authorization_endpoint: str
    token_endpoint: str
    resource: str | None = None
    registration_endpoint: str | None = None


async def discover(mcp_url: str) -> OAuthDiscovery:
    """
    Discover OAuth endpoints for an MCP server.

    RFC 9728 flow:
    1. POST initialize → expect 401 with WWW-Authenticate: Bearer resource_metadata=<url>
    2. GET resource_metadata → authorization_servers list
    3. GET /.well-known/oauth-authorization-server → endpoints

    Results are cached in-process for _CACHE_TTL seconds.
    """
    now = time.monotonic()
    if mcp_url in _cache:
        expires_at, cached = _cache[mcp_url]
        if now < expires_at:
            return cached

    resource_metadata_url = await _probe_resource_metadata(mcp_url)
    resource_metadata = await _get_json(resource_metadata_url)

    resource = resource_metadata.get("resource")
    auth_servers: list[str] = resource_metadata.get("authorization_servers", [])

    oauth_metadata: dict | None = None
    for server_url in auth_servers:
        oauth_metadata = await _get_oauth_server_metadata(server_url)
        if oauth_metadata:
            break

    if not oauth_metadata:
        raise ValueError(f"No OAuth metadata discoverable for MCP server: {mcp_url}")

    result = OAuthDiscovery(
        authorization_endpoint=oauth_metadata["authorization_endpoint"],
        token_endpoint=oauth_metadata["token_endpoint"],
        resource=resource,
        registration_endpoint=oauth_metadata.get("registration_endpoint"),
    )
    _cache[mcp_url] = (now + _CACHE_TTL, result)
    return result


async def _probe_resource_metadata(mcp_url: str) -> str:
    """POST initialize to trigger a 401 and extract the resource_metadata URL."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                mcp_url,
                json={"jsonrpc": "2.0", "method": "initialize", "params": {}, "id": 1},
            )
            if response.status_code == 401:
                www_auth = response.headers.get("WWW-Authenticate", "")
                match = re.search(r'resource_metadata="?([^",\s]+)"?', www_auth)
                if match:
                    return match.group(1)
    except httpx.HTTPError:
        pass

    parsed = urlparse(mcp_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    for path in (
        "/.well-known/oauth-protected-resource",
        "/.well-known/oauth-protected-resource.json",
    ):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(base + path)
                if response.status_code == 200:
                    return base + path
        except httpx.HTTPError:
            continue

    raise ValueError(f"Cannot discover resource metadata for {mcp_url}")


async def _get_json(url: str) -> dict:
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()


async def _get_oauth_server_metadata(auth_server_url: str) -> dict | None:
    base = auth_server_url.rstrip("/")
    for path in ("/.well-known/oauth-authorization-server", "/.well-known/openid-configuration"):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(base + path)
                if response.status_code == 200:
                    data = response.json()
                    if "authorization_endpoint" in data and "token_endpoint" in data:
                        return data
        except httpx.HTTPError:
            continue
    return None


def clear_cache() -> None:
    """Clear the discovery cache — useful in tests."""
    _cache.clear()
