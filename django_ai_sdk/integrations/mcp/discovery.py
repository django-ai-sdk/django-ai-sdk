from __future__ import annotations

import asyncio
import ipaddress
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import httpx
from mcp.client.auth.utils import (
    build_oauth_authorization_server_metadata_discovery_urls,
    build_protected_resource_metadata_discovery_urls,
    extract_resource_metadata_from_www_auth,
    handle_auth_metadata_response,
    handle_protected_resource_response,
)
from mcp.shared.auth_utils import check_resource_allowed

from django_ai_sdk.utils import resolve_setting

if TYPE_CHECKING:
    from mcp.shared.auth import OAuthMetadata, ProtectedResourceMetadata

logger = logging.getLogger(__name__)

_cache: dict[str, tuple[float, OAuthDiscovery]] = {}


def _cache_ttl() -> int:
    return resolve_setting("AI_SDK_MCP_DISCOVERY_CACHE_TTL", 3600)


def _discovery_timeout() -> int:
    return resolve_setting("AI_SDK_MCP_DISCOVERY_TIMEOUT", 10)


def _allowed_issuer_domains() -> list[str] | None:
    return resolve_setting("AI_SDK_MCP_ALLOWED_ISSUER_DOMAINS")


def _is_unsafe_ip(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True
    return ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local


async def _is_safe_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        hostname = parsed.hostname
        if not hostname:
            return False
        try:
            addrs = await asyncio.get_running_loop().getaddrinfo(hostname, None)
        except OSError:
            return False
        return not any(_is_unsafe_ip(str(addr[4][0])) for addr in addrs)
    except Exception:
        return False


async def _is_valid_issuer(issuer: str) -> bool:
    return await _is_safe_url(issuer)


@dataclass
class OAuthDiscovery:
    authorization_endpoint: str
    token_endpoint: str
    resource: str | None = None
    registration_endpoint: str | None = None


async def discover(
    mcp_url: str,
    expected_resource: str | None = None,
    allowed_issuer_domains: list[str] | None = None,
    use_cache: bool = True,
) -> OAuthDiscovery:
    now = time.monotonic()
    if use_cache and mcp_url in _cache:
        expires_at, cached = _cache[mcp_url]
        if now < expires_at:
            logger.debug("Discovery cache hit for %s", mcp_url)
            return cached

    if not await _is_safe_url(mcp_url):
        raise ValueError(f"Unsafe MCP URL: {mcp_url}")

    logger.info("Discovering OAuth for MCP server: %s", mcp_url)
    async with httpx.AsyncClient(timeout=_discovery_timeout(), follow_redirects=False) as client:
        prm = await _fetch_protected_resource_metadata(client, mcp_url)

        resource = str(prm.resource)
        if expected_resource and not check_resource_allowed(
            requested_resource=resource, configured_resource=expected_resource
        ):
            raise ValueError(f"Resource mismatch: expected {expected_resource}, got {resource}")

        auth_servers = [str(s) for s in prm.authorization_servers]
        logger.debug("Discovered %d authorization server(s) for %s", len(auth_servers), mcp_url)
        await _validate_issuers(auth_servers, allowed_issuer_domains)

        asm = await _fetch_authorization_server_metadata(client, auth_servers, mcp_url)

    result = OAuthDiscovery(
        authorization_endpoint=str(asm.authorization_endpoint),
        token_endpoint=str(asm.token_endpoint),
        resource=resource,
        registration_endpoint=str(asm.registration_endpoint) if asm.registration_endpoint else None,
    )

    if use_cache and _cache_ttl() > 0:
        _cache[mcp_url] = (now + _cache_ttl(), result)
        logger.debug("Cached discovery result for %s (TTL: %ds)", mcp_url, _cache_ttl())

    return result


async def _probe_www_auth(client: httpx.AsyncClient, mcp_url: str) -> str | None:
    try:
        response = await client.post(
            mcp_url,
            json={"jsonrpc": "2.0", "method": "initialize", "params": {}, "id": 1},
        )
    except httpx.HTTPError as e:
        logger.debug("Probe for resource metadata failed: %s", e)
        return None
    url = extract_resource_metadata_from_www_auth(response)
    if url and await _is_safe_url(url):
        return url
    return None


async def _fetch_protected_resource_metadata(
    client: httpx.AsyncClient, mcp_url: str
) -> ProtectedResourceMetadata:
    www_auth_url = await _probe_www_auth(client, mcp_url)
    for url in build_protected_resource_metadata_discovery_urls(www_auth_url, mcp_url):
        if not await _is_safe_url(url):
            continue
        try:
            response = await client.get(url)
        except httpx.HTTPError as e:
            logger.debug("Protected-resource metadata fetch failed for %s: %s", url, e)
            continue
        prm = await handle_protected_resource_response(response)
        if prm is not None:
            return prm
    raise ValueError(f"Cannot discover resource metadata for {mcp_url}")


async def _fetch_authorization_server_metadata(
    client: httpx.AsyncClient, auth_servers: list[str], mcp_url: str
) -> OAuthMetadata:
    failed_issuers: list[str] = []
    for server_url in auth_servers:
        for url in build_oauth_authorization_server_metadata_discovery_urls(server_url, mcp_url):
            if not await _is_safe_url(url):
                continue
            try:
                response = await client.get(url)
            except httpx.HTTPError as e:
                logger.debug("Auth-server metadata fetch failed for %s: %s", url, e)
                continue
            keep_trying, asm = await handle_auth_metadata_response(response)
            if asm is not None:
                logger.info("Successfully discovered OAuth metadata from %s", server_url)
                return asm
            if not keep_trying:
                break
        failed_issuers.append(server_url)

    raise ValueError(
        f"No valid OAuth metadata found. Tried {len(auth_servers)} issuer(s). "
        f"Failed: {failed_issuers}"
    )


async def _validate_issuers(
    auth_servers: list[str], allowed_issuer_domains: list[str] | None
) -> None:
    domains_to_check = (
        allowed_issuer_domains if allowed_issuer_domains is not None else _allowed_issuer_domains()
    )
    for issuer in auth_servers:
        if not await _is_valid_issuer(issuer):
            raise ValueError(f"Invalid issuer format: {issuer}")

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


def clear_cache() -> None:
    _cache.clear()
