from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


class StaticMCPServer(BaseModel):
    type: Literal["static"] = "static"
    url: str
    label: str = ""
    tools: list[str] = []


class TokenMCPServer(BaseModel):
    type: Literal["token"] = "token"
    url: str
    label: str = ""
    token: str
    tools: list[str] = []

    @model_validator(mode="after")
    def validate_token(self) -> TokenMCPServer:
        if not self.token:
            raise ValueError("token must not be empty for TokenMCPServer")
        return self


class OAuthMCPServer(BaseModel):
    type: Literal["oauth"] = "oauth"
    url: str
    label: str = ""
    client_id: str = ""
    client_secret: str = ""
    scope: str = ""
    oauth_discovery_url: str = ""
    authorization_endpoint: str = ""
    token_endpoint: str = ""
    tools: list[str] = []

    @model_validator(mode="after")
    def validate_endpoints(self) -> OAuthMCPServer:
        has_auth = bool(self.authorization_endpoint)
        has_token = bool(self.token_endpoint)
        if has_auth != has_token:
            raise ValueError(
                "authorization_endpoint and token_endpoint must both be set or both be empty"
            )
        return self


MCPServer = Annotated[
    StaticMCPServer | TokenMCPServer | OAuthMCPServer,
    Field(discriminator="type"),
]


class ConnectionOut(BaseModel):
    """MCP server connection status."""

    server_name: str
    label: str
    type: str
    connected: bool | None = None


class AssistantMCPServerStatus(BaseModel):
    """MCP server status for an assistant."""

    server_name: str
    label: str
    type: str
    status: str  # "active", "expired", "disconnected"
    tool_names: list[str]
