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
