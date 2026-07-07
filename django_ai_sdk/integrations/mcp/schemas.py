from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, SecretStr, model_validator

from django_ai_sdk.integrations.base import IntegrationStatus


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
    StaticMCPIntegrationConfig | TokenMCPIntegrationConfig | OAuthMCPIntegrationConfig,
    Field(discriminator="type"),
]


class ConnectionOut(BaseModel):
    """MCP server connection status (staff config UI: which servers exist to pick from)."""

    server_name: str
    label: str
    type: str
    connected: bool | None = None
    has_token: bool = False
    status: IntegrationStatus


class AssistantIntegrationStatus(BaseModel):
    """One configured integration's status for a given assistant/user."""

    server_name: str
    label: str
    type: str
    status: IntegrationStatus
    tool_names: list[str]
