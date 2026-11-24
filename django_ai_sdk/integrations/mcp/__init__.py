from __future__ import annotations

from django_ai_sdk.integrations.mcp.schemas import (
    # A union type alias, not a class: never isinstance() against it.
    MCPIntegrationConfig,
    OAuthMCPIntegrationConfig,
    StaticMCPIntegrationConfig,
    TokenMCPIntegrationConfig,
)

__all__ = [
    "MCPIntegrationConfig",
    "OAuthMCPIntegrationConfig",
    "StaticMCPIntegrationConfig",
    "TokenMCPIntegrationConfig",
]
