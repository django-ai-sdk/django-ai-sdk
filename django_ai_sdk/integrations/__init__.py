"""Integrations: the SDK's uniform capability layer for pluggable third-party
functionality, such as MCP servers, hand-written API wrappers, and anything added
later.

Every integration is one Integration (the contract in base.py), living in its own
Django app. Its AppConfig (subclassing IntegrationAppConfig) registers it on
ready(). The assistant, the /api/integrations router, and the status endpoints all
talk to the one Integration contract, so none of them know or care which kind an
integration is.

Package layout, two tiers:

- Framework (this directory's top-level modules): base.py (the contract plus
  ResilientCache), registry.py, secrets.py (per-integration secret resolution),
  and apps.py. Nothing here knows about MCP or any specific provider.
- Kinds (api/, mcp/): base classes and shared infra for a category of integration.
  api.base.APIIntegration is the base for hand-written API wrappers. mcp/ is the MCP
  toolkit (OAuth token models, discovery/PKCE, the Haystack MCPToolset loader) for
  any MCP server. Both kinds produce ordinary Haystack tool objects, consumed
  identically by the assistant.

Resilience is part of the contract, not something each integration reimplements: tool
lists are cached stale-while-revalidate behind a per-key circuit breaker, so a slow or
dead integration costs a bounded one-off wait and then nothing, and reports itself as
DEGRADED instead of silently contributing no tools.
"""

from __future__ import annotations

from django_ai_sdk.integrations.api.base import APIIntegration
from django_ai_sdk.integrations.apps import IntegrationAppConfig
from django_ai_sdk.integrations.base import (
    Integration,
    IntegrationNotConnectable,
    IntegrationStatus,
)

__all__ = [
    "APIIntegration",
    "IntegrationAppConfig",
    "IntegrationNotConnectable",
    "Integration",
    "IntegrationStatus",
]
