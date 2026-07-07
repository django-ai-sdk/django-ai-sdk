"""SSRF-guard regression tests for MCP OAuth discovery (RFC 9728).

`_is_safe_url` is the one gate standing between a (possibly compromised) MCP
server's own response — e.g. its `authorization_servers` list — and an outbound
request from this process. These tests exist because the original implementation
only rejected *literal* private/loopback/reserved IPs, silently allowing a
hostname that DNS-resolves to one of those (the classic SSRF-via-DNS-rebinding
gap) straight through.
"""

from __future__ import annotations

import asyncio

from django_ai_sdk.integrations.mcp.discovery import _is_safe_url, _is_valid_issuer


def _mock_resolver(monkeypatch, ip: str) -> None:
    async def fake_getaddrinfo(host, port):
        return [(None, None, None, None, (ip, 0))]

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", fake_getaddrinfo)


class TestIsSafeUrl:
    async def test_rejects_non_http_scheme(self):
        assert await _is_safe_url("ftp://example.com") is False

    async def test_rejects_url_with_no_hostname(self):
        assert await _is_safe_url("http://") is False

    async def test_rejects_literal_loopback_ip(self):
        assert await _is_safe_url("http://127.0.0.1/mcp") is False

    async def test_rejects_literal_link_local_metadata_ip(self):
        assert await _is_safe_url("http://169.254.169.254/latest/meta-data/") is False

    async def test_rejects_literal_private_ip(self):
        assert await _is_safe_url("http://10.0.0.5/mcp") is False

    async def test_accepts_literal_public_ip(self):
        assert await _is_safe_url("http://93.184.216.34/mcp") is True

    async def test_rejects_hostname_that_resolves_to_loopback(self, monkeypatch):
        """The actual bug: a hostname isn't safe just because it isn't itself an
        IP literal — it can still resolve to 127.0.0.1 via DNS."""
        _mock_resolver(monkeypatch, "127.0.0.1")
        assert await _is_safe_url("http://attacker.example.com/mcp") is False

    async def test_rejects_hostname_that_resolves_to_metadata_ip(self, monkeypatch):
        _mock_resolver(monkeypatch, "169.254.169.254")
        assert await _is_safe_url("http://attacker.example.com/mcp") is False

    async def test_accepts_hostname_that_resolves_to_public_ip(self, monkeypatch):
        _mock_resolver(monkeypatch, "93.184.216.34")
        assert await _is_safe_url("http://mcp.example.com/mcp") is True

    async def test_rejects_on_dns_resolution_failure(self, monkeypatch):
        """Fails closed rather than open when the hostname can't be resolved."""

        async def fake_getaddrinfo(host, port):
            raise OSError("nodename nor servname provided, or not known")

        monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", fake_getaddrinfo)
        assert await _is_safe_url("http://does-not-resolve.invalid/mcp") is False


class TestIsValidIssuer:
    async def test_delegates_to_is_safe_url(self, monkeypatch):
        _mock_resolver(monkeypatch, "127.0.0.1")
        assert await _is_valid_issuer("https://attacker.example.com") is False

    async def test_valid_public_issuer_is_accepted(self, monkeypatch):
        _mock_resolver(monkeypatch, "93.184.216.34")
        assert await _is_valid_issuer("https://accounts.example.com") is True
