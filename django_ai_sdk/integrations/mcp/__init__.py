from __future__ import annotations

from django_ai_sdk.integrations.mcp.schemas import (
    # NB: MCPIntegrationConfig is a discriminated-union type alias
    # (Annotated[Static | Token | UserToken | OAuth, ...]), not a class — don't
    # `isinstance()` against it. Use it only as a type annotation for an MCP config value.
    MCPIntegrationConfig,
    OAuthMCPIntegrationConfig,
    StaticMCPIntegrationConfig,
    TokenMCPIntegrationConfig,
    UserTokenMCPIntegrationConfig,
)

__all__ = [
    "MCPIntegrationConfig",
    "OAuthMCPIntegrationConfig",
    "StaticMCPIntegrationConfig",
    "TokenMCPIntegrationConfig",
    "UserTokenMCPIntegrationConfig",
]
