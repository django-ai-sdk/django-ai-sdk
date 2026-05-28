"""RFC 9728 — OAuth 2.0 Protected Resource Metadata discovery for MCP servers."""

from __future__ import annotations

import ipaddress
import logging
import re
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
from django.conf import settings

logger = logging.getLogger(__name__)

_cache: dict[str, tuple[float, OAuthDiscovery]] = {}


def _cache_ttl() -> int:
    return getattr(settings, "AI_SDK_MCP_DISCOVERY_CACHE_TTL", 3600)


def _discovery_timeout() -> int:
    return getattr(settings, "AI_SDK_MCP_DISCOVERY_TIMEOUT", 10)


def _allowed_issuer_domains() -> list[str] | None:
    return getattr(settings, "AI_SDK_MCP_ALLOWED_ISSUER_DOMAINS", None)


def _is_safe_url(url: str) -> bool:
    """Validate URL is http(s) and not a private/reserved IP (SSRF protection)."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        hostname = parsed.hostname
        if not hostname:
            return False
        # Reject private and reserved IP ranges (loopback, private networks, metadata services, etc)
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local:
                return False
        except ValueError:
            # Not an IP, hostname is OK
            pass
        return True
    except Exception:
        return False


def _is_valid_issuer(issuer: str) -> bool:
    """Validate issuer is a proper URI per RFC 9728 (authorization_servers field)."""
    return _is_safe_url(issuer)


@dataclass
class OAuthDiscovery:
    authorization_endpoint: str
    token_endpoint: str
    resource: str | None = None
    registration_endpoint: str | None = None
    scopes_supported: list[str] | None = None
    bearer_methods_supported: list[str] | None = None


async def discover(
    mcp_url: str,
    expected_resource: str | None = None,
    allowed_issuer_domains: list[str] | None = None,
    use_cache: bool = True,
) -> OAuthDiscovery:
    """
    Discover OAuth endpoints for an MCP server per RFC 9728.

    RFC 9728 flow:
    1. POST initialize → expect 401 with WWW-Authenticate: Bearer resource_metadata=<url>
    2. GET resource_metadata → authorization_servers list, validate resource
    3. GET /.well-known/oauth-authorization-server → endpoints

    Args:
        mcp_url: The MCP server URL to discover OAuth for
        expected_resource: Optional. If provided, validate the discovered resource matches
        allowed_issuer_domains: Optional. List of allowed OAuth issuer domains (e.g., ["accounts.notion.com"])
                               Enables defense-in-depth against compromised MCP servers.
                               - None (default): No restriction, allow any issuer
                               - [] (empty): Reject all issuers
                               - ["domain"]: Only allow specified domains
                               If not provided, uses AI_SDK_MCP_ALLOWED_ISSUER_DOMAINS from settings.
        use_cache: Whether to use cached results. Default True.

    Results are cached in-process for AI_SDK_MCP_DISCOVERY_CACHE_TTL seconds (default 3600).

    Settings (configure in Django settings.py):
        AI_SDK_MCP_DISCOVERY_CACHE_TTL: Cache TTL in seconds (default 3600)
        AI_SDK_MCP_DISCOVERY_TIMEOUT: HTTP request timeout in seconds (default 10)
        AI_SDK_MCP_ALLOWED_ISSUER_DOMAINS: Default list of allowed OAuth issuer domains

    Raises:
        ValueError: If discovery fails, resource validation fails, or URL is unsafe
    """
    now = time.monotonic()
    if use_cache and mcp_url in _cache:
        expires_at, cached = _cache[mcp_url]
        if now < expires_at:
            logger.debug("Discovery cache hit for %s", mcp_url)
            return cached

    if not _is_safe_url(mcp_url):
        raise ValueError(f"Unsafe MCP URL: {mcp_url}")

    logger.info("Discovering OAuth for MCP server: %s", mcp_url)
    resource_metadata_url = await _probe_resource_metadata(mcp_url)
    resource_metadata = await _get_json(resource_metadata_url)

    resource = resource_metadata.get("resource")
    if not resource:
        raise ValueError(f"Missing 'resource' in metadata from {resource_metadata_url}")

    # RFC 9728: Client should validate resource matches expectations
    if expected_resource and resource != expected_resource:
        raise ValueError(f"Resource mismatch: expected {expected_resource}, got {resource}")

    auth_servers: list[str] = resource_metadata.get("authorization_servers", [])
    if not auth_servers:
        raise ValueError(f"No authorization_servers in metadata from {resource_metadata_url}")

    logger.debug("Discovered %d authorization server(s) for %s", len(auth_servers), mcp_url)

    # Use settings default if not explicitly provided
    domains_to_check = (
        allowed_issuer_domains if allowed_issuer_domains is not None else _allowed_issuer_domains()
    )

    # Validate issuer format and allowed domains
    for issuer in auth_servers:
        if not _is_valid_issuer(issuer):
            raise ValueError(f"Invalid issuer format: {issuer}")

        # Check against allowlist if configured (None = no restriction, [] = allow none, [...] = allow specific)
        # Each entry may be a bare hostname ("mcp.notion.com") or a full URL
        # ("https://mcp.notion.com/mcp") — normalize both to netloc for comparison.
        if domains_to_check is not None:
            issuer_domain = urlparse(issuer).netloc
            normalized = {urlparse(d).netloc if "://" in d else d for d in domains_to_check}
            if issuer_domain not in normalized:
                logger.warning(
                    "Issuer %s not in allowed domains %s (defense-in-depth check)",
                    issuer_domain,
                    domains_to_check,
                )
                raise ValueError(f"Issuer {issuer_domain} not in allowed domains")

    oauth_metadata: dict | None = None
    failed_issuers = []
    for server_url in auth_servers:
        try:
            oauth_metadata = await _get_oauth_server_metadata(server_url)
            if oauth_metadata:
                logger.info("Successfully discovered OAuth metadata from %s", server_url)
                break
        except Exception as e:
            logger.debug("Failed to get metadata from %s: %s", server_url, e)
            failed_issuers.append(server_url)

    if not oauth_metadata:
        raise ValueError(
            f"No valid OAuth metadata found. Tried {len(auth_servers)} issuer(s). "
            f"Failed: {failed_issuers}"
        )

    result = OAuthDiscovery(
        authorization_endpoint=oauth_metadata["authorization_endpoint"],
        token_endpoint=oauth_metadata["token_endpoint"],
        resource=resource,
        registration_endpoint=oauth_metadata.get("registration_endpoint"),
        scopes_supported=oauth_metadata.get("scopes_supported"),
        bearer_methods_supported=oauth_metadata.get("bearer_methods_supported"),
    )

    if use_cache and _cache_ttl() > 0:
        _cache[mcp_url] = (now + _cache_ttl(), result)
        logger.debug("Cached discovery result for %s (TTL: %ds)", mcp_url, _cache_ttl())

    return result


async def _probe_resource_metadata(mcp_url: str) -> str:
    """POST initialize to trigger a 401 and extract the resource_metadata URL per RFC 9728."""
    try:
        async with httpx.AsyncClient(
            timeout=_discovery_timeout(), follow_redirects=False
        ) as client:
            response = await client.post(
                mcp_url,
                json={"jsonrpc": "2.0", "method": "initialize", "params": {}, "id": 1},
            )
            if response.status_code == 401:
                www_auth = response.headers.get("WWW-Authenticate", "")
                match = re.search(r'resource_metadata="?([^",\s]+)"?', www_auth)
                if match:
                    url = match.group(1)
                    if _is_safe_url(url):
                        return url
    except httpx.HTTPError as e:
        logger.debug("Probe for resource metadata failed: %s", e)

    parsed = urlparse(mcp_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    # RFC 9728 specifies this well-known path
    try:
        async with httpx.AsyncClient(
            timeout=_discovery_timeout(), follow_redirects=False
        ) as client:
            response = await client.get(base + "/.well-known/oauth-protected-resource")
            if response.status_code == 200:
                return base + "/.well-known/oauth-protected-resource"
    except httpx.HTTPError as e:
        logger.debug("Well-known resource metadata endpoint not found: %s", e)

    raise ValueError(f"Cannot discover resource metadata for {mcp_url}")


async def _get_json(url: str) -> dict:
    """Safely fetch and parse JSON from a URL with security checks."""
    if not _is_safe_url(url):
        raise ValueError(f"Unsafe URL: {url}")

    async with httpx.AsyncClient(timeout=_discovery_timeout(), follow_redirects=False) as client:
        response = await client.get(url)
        response.raise_for_status()

        # Validate content-type
        content_type = response.headers.get("content-type", "").lower()
        if "application/json" not in content_type:
            raise ValueError(f"Invalid content-type from {url}: {content_type}")

        return response.json()


async def _get_oauth_server_metadata(auth_server_url: str) -> dict | None:
    """Fetch OAuth authorization server metadata from well-known endpoints."""
    if not _is_valid_issuer(auth_server_url):
        return None

    base = auth_server_url.rstrip("/")
    for path in ("/.well-known/oauth-authorization-server", "/.well-known/openid-configuration"):
        try:
            async with httpx.AsyncClient(
                timeout=_discovery_timeout(), follow_redirects=False
            ) as client:
                response = await client.get(base + path)
                if response.status_code == 200:
                    content_type = response.headers.get("content-type", "").lower()
                    if "application/json" not in content_type:
                        continue
                    data = response.json()
                    if "authorization_endpoint" in data and "token_endpoint" in data:
                        return data
        except httpx.HTTPError:
            continue
    return None


def clear_cache() -> None:
    """Clear the discovery cache — useful in tests."""
    _cache.clear()
