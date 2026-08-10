from __future__ import annotations

import base64
import hashlib
import ipaddress
import logging
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlparse

from cryptography.fernet import Fernet, InvalidToken
from django.conf import settings
from django.db import models
from django.utils import timezone as tz

logger = logging.getLogger(__name__)

_BLOCKED_HOSTNAMES = {"localhost", "metadata.google.internal"}
_BLOCKED_HOSTNAME_SUFFIXES = (".local", ".internal", ".localhost")


def _validate_public_url(url: str) -> None:
    """Reject a URL an SSRF-minded staff account could use to reach this app's own
    network (a literal loopback/private/link-local IP, or a known internal
    hostname) -- this app's own process is what fetches MCPServerConfig.url.

    Static checks only, no DNS resolution: a hostname that *resolves* to a private
    address isn't caught here (egress-firewall the deployment for that guarantee).
    A hostname is also how a deployment that genuinely wants an internal MCP server
    gets one -- no override setting needed for that.
    """
    host = (urlparse(url).hostname or "").lower()
    if host in _BLOCKED_HOSTNAMES or host.endswith(_BLOCKED_HOSTNAME_SUFFIXES):
        raise ValueError(f"{url!r} points at a local/internal hostname ({host!r})")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
        raise ValueError(f"{url!r} resolves to a non-public address ({ip})")


def _get_fernet() -> Fernet:
    """Derive the Fernet key used to encrypt/decrypt integration credentials
    from Django's SECRET_KEY, so there's no second secret to configure."""
    secret = settings.SECRET_KEY
    if isinstance(secret, str):
        secret = secret.encode()
    key = base64.urlsafe_b64encode(hashlib.sha256(secret).digest())
    return Fernet(key)


class MCPOAuthToken(models.Model):
    user_id: int
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="mcp_oauth_tokens",
    )
    server_name = models.CharField(max_length=100)
    access_token = models.TextField()
    refresh_token = models.TextField(blank=True)
    token_type = models.CharField(max_length=50, default="Bearer")
    expires_at = models.DateTimeField(null=True, blank=True)
    scope = models.TextField(blank=True)

    class Meta:
        app_label = "django_ai_sdk_mcp"
        unique_together = ("user", "server_name")

    def __str__(self) -> str:
        return f"{self.user}:{self.server_name}"

    def set_tokens(self, token_response: dict[str, Any]) -> None:
        f = _get_fernet()
        if access := token_response.get("access_token"):
            self.access_token = f.encrypt(access.encode()).decode()
        else:
            logger.warning("set_tokens called without access_token for %s", self)
        if refresh := token_response.get("refresh_token"):
            self.refresh_token = f.encrypt(refresh.encode()).decode()
        self.token_type = token_response.get("token_type", "Bearer")
        self.scope = token_response.get("scope", "")
        if expires_in := token_response.get("expires_in"):
            self.expires_at = tz.now() + timedelta(seconds=int(expires_in))
        elif ts := token_response.get("expires_at"):
            if isinstance(ts, (int, float)):
                self.expires_at = datetime.fromtimestamp(ts, tz=UTC)

    def get_access_token(self) -> str:
        if not self.access_token:
            return ""
        try:
            return _get_fernet().decrypt(self.access_token.encode()).decode()
        except InvalidToken:
            logger.error("Failed to decrypt access token for %s", self)
            return ""

    def get_refresh_token(self) -> str:
        if not self.refresh_token:
            return ""
        try:
            return _get_fernet().decrypt(self.refresh_token.encode()).decode()
        except InvalidToken:
            logger.error("Failed to decrypt refresh token for %s", self)
            return ""

    def is_expired(self) -> bool:
        return bool(self.expires_at and self.expires_at <= tz.now())


class MCPOAuthClient(models.Model):
    """Stores dynamically-registered OAuth client credentials (RFC 7591) per MCP server."""

    server_name = models.CharField(max_length=100, primary_key=True)
    client_id = models.CharField(max_length=500)
    client_secret = models.TextField(blank=True)
    redirect_uri = models.URLField()
    registered_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = "django_ai_sdk_mcp"

    def __str__(self) -> str:
        return f"OAuthClient({self.server_name})"

    def set_credentials(self, client_id: str, client_secret: str) -> None:
        """Store client credentials; client_secret is encrypted, client_id is plaintext."""
        f = _get_fernet()
        self.client_id = client_id
        if client_secret:
            self.client_secret = f.encrypt(client_secret.encode()).decode()
        else:
            self.client_secret = ""

    def get_client_secret(self) -> str:
        """Retrieve decrypted client_secret."""
        if not self.client_secret:
            return ""
        try:
            return _get_fernet().decrypt(self.client_secret.encode()).decode()
        except InvalidToken:
            logger.error("Failed to decrypt client_secret for %s", self)
            return ""


class MCPServerConfig(models.Model):
    """One MCP server declared as data instead of code -- add/edit/enable from
    Django admin or a settings UI, no app, no deploy, no restart.

    Never overrides a same-named installed-app integration (see
    integrations.registry); this is for adding a server with no code at all.

    ``token``/``client_secret`` are Fernet-encrypted, same as ``MCPOAuthToken``.
    Auth kinds match ``mcp.loader.AUTH_KINDS``.
    """

    AUTH_CHOICES = [
        ("static", "No auth"),
        ("token", "Shared token"),
        ("oauth", "OAuth 2.1"),
    ]

    name = models.SlugField(
        unique=True, help_text="Registry key, e.g. 'zendesk'. Must be URL-safe."
    )
    label = models.CharField(max_length=200, blank=True)
    hint = models.TextField(
        blank=True,
        help_text=(
            "What this server's data actually is, e.g. 'Company wiki, HR docs, and "
            "engineering runbooks.' Prepended to every tool's description so the "
            "model knows when to reach for it, not just that it exists."
        ),
    )
    url = models.URLField()
    auth = models.CharField(max_length=20, choices=AUTH_CHOICES, default="static")
    token = models.TextField(blank=True, help_text="Encrypted. Only used when auth='token'.")
    client_id = models.CharField(max_length=500, blank=True)
    client_secret = models.TextField(
        blank=True, help_text="Encrypted. Only used when auth='oauth'."
    )
    scope = models.CharField(max_length=500, blank=True)
    oauth_discovery_url = models.URLField(blank=True)
    authorization_endpoint = models.URLField(blank=True)
    token_endpoint = models.URLField(blank=True)
    tools = models.JSONField(default=list, blank=True, help_text="Tool allow-list; empty = all.")
    enabled = models.BooleanField(
        default=True, help_text="Unchecking removes it from the registry immediately."
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = "django_ai_sdk_mcp"
        verbose_name = "MCP server"

    def __str__(self) -> str:
        return self.name

    def save(self, *args: Any, **kwargs: Any) -> None:
        _validate_public_url(self.url)
        super().save(*args, **kwargs)

    def set_token(self, token: str) -> None:
        self.token = _get_fernet().encrypt(token.encode()).decode() if token else ""

    def get_token(self) -> str:
        if not self.token:
            return ""
        try:
            return _get_fernet().decrypt(self.token.encode()).decode()
        except InvalidToken:
            logger.error("Failed to decrypt token for MCPServerConfig %r", self.name)
            return ""

    def set_client_secret(self, client_secret: str) -> None:
        self.client_secret = (
            _get_fernet().encrypt(client_secret.encode()).decode() if client_secret else ""
        )

    def get_client_secret(self) -> str:
        if not self.client_secret:
            return ""
        try:
            return _get_fernet().decrypt(self.client_secret.encode()).decode()
        except InvalidToken:
            logger.error("Failed to decrypt client_secret for MCPServerConfig %r", self.name)
            return ""

    def to_integration(self) -> Any:
        """Build the ``DynamicMCPIntegration`` this row describes.

        Never raises -- a bad row (missing url, unknown auth) registers as needing
        setup instead of breaking the registry for every other integration. Imports
        are local to avoid a module-level cycle with loader.py, which already
        imports this module's OAuth models the same way.
        """
        from django_ai_sdk.integrations.mcp.loader import (
            DynamicMCPIntegration,
            build_mcp_config_safe,
        )

        config, needs_setup = build_mcp_config_safe(
            auth=self.auth,
            url=self.url,
            label=self.label or self.name.title(),
            hint=self.hint,
            tools=list(self.tools or []),
            scope=self.scope,
            client_id=self.client_id,
            client_secret=self.get_client_secret(),
            oauth_discovery_url=self.oauth_discovery_url,
            authorization_endpoint=self.authorization_endpoint,
            token_endpoint=self.token_endpoint,
            token=self.get_token(),
        )
        return DynamicMCPIntegration(
            self.name, config, needs_setup=needs_setup, intended_kind=self.auth
        )
