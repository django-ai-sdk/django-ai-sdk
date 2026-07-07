from __future__ import annotations

from django_ai_sdk.integrations.mcp.schemas import (
    # NB: MCPIntegrationConfig is a discriminated-union type alias
    # (Annotated[Static | Token | OAuth, ...]), not a class — don't `isinstance()`
    # against it. Use it only as a type annotation for AI_SDK_INTEGRATIONS values.
    MCPIntegrationConfig,
    OAuthMCPIntegrationConfig,
    StaticMCPIntegrationConfig,
    TokenMCPIntegrationConfig,
)

__all__ = [
    "MCPIntegrationConfig",
    "StaticMCPIntegrationConfig",
    "TokenMCPIntegrationConfig",
    "OAuthMCPIntegrationConfig",
]
