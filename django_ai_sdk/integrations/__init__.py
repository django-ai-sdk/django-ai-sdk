"""Integrations: the SDK's uniform capability layer for pluggable third-party
functionality — MCP servers, hand-written API wrappers, and anything added later.

Every integration is one ``Integration`` (the contract in ``base.py``), declared
in ``AI_SDK_INTEGRATIONS`` as ``{name: "dotted.path.To.Service"}``. The assistant, the
``/api/integrations`` router and the status endpoints all talk to that one contract, so
none of them know or care which kind an integration is.

Package layout — three tiers:

- **Framework** (this directory's top-level modules): ``base.py`` (the contract plus
  ``ResilientCache``), ``registry.py``, ``views.py``/``schemas.py`` (the kind-agnostic
  ``/api/integrations`` router), and ``apps.py`` (the optional Django-app escape hatch
  for an integration that ships models or admin). Nothing here knows about MCP or any
  specific provider.
- **Kinds** (``api/``, ``mcp/``): base classes and shared infra for a category of
  integration. ``api.base.APIIntegration`` is the base for hand-written API wrappers.
  ``mcp/`` is the MCP toolkit — OAuth token models, discovery/PKCE, the Haystack
  ``MCPToolset`` loader — for any MCP server. Both kinds produce ordinary Haystack tool
  objects, consumed identically by the assistant: MCP is, in the end, just another way
  to get tools.
- **Defaults** (``defaults.py``): ready-made services for common third-party products
  (Notion, Linear) so no project hand-rolls that config. Your own integrations live in
  your own app — see the demo's ``weather`` app as a template.

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
