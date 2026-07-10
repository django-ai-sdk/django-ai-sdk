from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, SecretStr, model_validator


class StaticMCPIntegrationConfig(BaseModel):
    type: Literal["static"] = "static"
    url: str
    label: str = ""
    tools: list[str] = []


class TokenMCPIntegrationConfig(BaseModel):
    type: Literal["token"] = "token"
    url: str
    label: str = ""
    token: SecretStr
    tools: list[str] = []

    @model_validator(mode="after")
    def validate_token(self) -> TokenMCPIntegrationConfig:
        if not self.token.get_secret_value():
            raise ValueError("token must not be empty for TokenMCPIntegrationConfig")
        return self


class UserTokenMCPIntegrationConfig(BaseModel):
    """A token-auth MCP server where each user supplies their own token.

    Unlike ``TokenMCPIntegrationConfig`` (one shared deployment secret), no token is
    configured here — it's submitted per-user via ``POST /api/integrations/<name>/
    credential`` and stored like an OAuth token (access_token only, no refresh/expiry).
    """

    type: Literal["user_token"] = "user_token"
    url: str
    label: str = ""
    tools: list[str] = []


class OAuthMCPIntegrationConfig(BaseModel):
    type: Literal["oauth"] = "oauth"
    url: str
    label: str = ""
    client_id: str = ""
    client_secret: SecretStr = SecretStr("")
    scope: str = ""
    oauth_discovery_url: str = ""
    authorization_endpoint: str = ""
    token_endpoint: str = ""
    tools: list[str] = []

    @model_validator(mode="after")
    def validate_endpoints(self) -> OAuthMCPIntegrationConfig:
        has_auth = bool(self.authorization_endpoint)
        has_token = bool(self.token_endpoint)
        if has_auth != has_token:
            raise ValueError(
                "authorization_endpoint and token_endpoint must both be set or both be empty"
            )
        return self


MCPIntegrationConfig = Annotated[
    StaticMCPIntegrationConfig
    | TokenMCPIntegrationConfig
    | UserTokenMCPIntegrationConfig
    | OAuthMCPIntegrationConfig,
    Field(discriminator="type"),
]
