from __future__ import annotations

import base64
import hashlib
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from django.conf import settings
from django.db import models
from django.utils import timezone as tz

logger = logging.getLogger(__name__)


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


class MCPServerConfig(models.Model):
    """An MCP server declared as data — added/edited/enabled via Django admin.

    A server declared here needs no app, no deploy, no restart: the registry (see
    ``integrations.registry``) picks it up on next access and drops it the moment
    ``enabled`` is unchecked or the row is deleted.

    ``token``/``client_secret`` are Fernet-encrypted, same as ``MCPOAuthToken``. For
    ``auth="user_token"`` no server-wide secret is stored here at all — each user
    submits their own via ``POST /api/integrations/<name>/credential``.
    """

    AUTH_CHOICES = [
        ("static", "No auth"),
        ("token", "Shared token"),
        ("user_token", "Per-user token"),
        ("oauth", "OAuth 2.1"),
    ]

    name = models.SlugField(unique=True, help_text="Registry key — must be URL-safe.")
    label = models.CharField(max_length=200, blank=True)
    url = models.URLField()
    auth = models.CharField(max_length=20, choices=AUTH_CHOICES, default="static")
    token = models.TextField(blank=True, help_text="Encrypted. Only used when auth='token'.")
    client_id = models.CharField(max_length=500, blank=True)
    client_secret = models.TextField(blank=True, help_text="Encrypted.")
    scope = models.CharField(max_length=500, blank=True)
    oauth_discovery_url = models.URLField(blank=True)
    authorization_endpoint = models.URLField(blank=True)
    token_endpoint = models.URLField(blank=True)
    tools = models.JSONField(default=list, blank=True, help_text="Tool allow-list; empty = all.")
    enabled = models.BooleanField(default=True, help_text="Unchecking removes it immediately.")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = "django_ai_sdk_mcp"
        verbose_name = "MCP server"

    def __str__(self) -> str:
        return self.name

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

    def to_config(self) -> tuple[Any, str | None]:
        """Build the pydantic MCP config for this row. Returns ``(config, needs_setup)``
        — never raises; see ``mcp.loader.build_mcp_config_safe``."""
        from django_ai_sdk.integrations.mcp.loader import build_mcp_config_safe

        return build_mcp_config_safe(
            auth=self.auth,
            url=self.url,
            label=self.label or self.name.title(),
            tools=list(self.tools or []),
            scope=self.scope,
            client_id=self.client_id,
            client_secret=self.get_client_secret(),
            oauth_discovery_url=self.oauth_discovery_url,
            authorization_endpoint=self.authorization_endpoint,
            token_endpoint=self.token_endpoint,
            token=self.get_token(),
        )


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
