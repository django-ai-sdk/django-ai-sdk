"""Integrations: the SDK's uniform capability layer for pluggable third-party
functionality — MCP servers, hand-written API wrappers, and anything added later.

Each integration is a Django app (see ``IntegrationAppConfig``) exposing one
``IntegrationService`` (the shared contract in ``base.py``).

Package layout — three tiers:

- **Framework** (this directory's top-level modules): ``apps.py``/``base.py``/
  ``registry.py``/``views.py``/``schemas.py`` — the ``IntegrationService`` contract,
  the process registry, and the kind-agnostic ``/api/integrations`` router. Nothing
  here knows about MCP or any specific provider.
- **Kinds** (``api/``, ``mcp/``): base classes and shared infra for a category of
  integration. ``api.base.APIIntegration`` is the base for hand-written API wrappers.
  ``mcp/`` is the MCP toolkit — models, OAuth discovery/PKCE, the Haystack
  ``MCPToolset`` loader — for any MCP server. Both kinds produce ordinary Haystack
  tool objects, consumed identically by the assistant: MCP is, in the end, just
  another way to get tools.
- **Defaults** (``defaults/``): integrations for common third-party services that
  the SDK maintains and ships best-practice config for, so no app has to hand-roll
  its own Linear/Notion setup — e.g. ``defaults/notion`` and ``defaults/linear``
  subclass ``mcp.MCPIntegrationService`` with a few lines of config. Enable one by
  adding its app to ``INSTALLED_APPS``. Your own integrations (custom APIs, MCP
  servers specific to your product) live in your own app instead — see the demo's
  ``weather``/``github`` apps as templates — not inside the SDK's ``defaults/``.
"""

from __future__ import annotations

from django_ai_sdk.integrations.api.base import APIIntegration
from django_ai_sdk.integrations.apps import IntegrationAppConfig
from django_ai_sdk.integrations.base import (
    IntegrationNotConnectable,
    IntegrationService,
    IntegrationStatus,
)

__all__ = [
    "APIIntegration",
    "IntegrationAppConfig",
    "IntegrationNotConnectable",
    "IntegrationService",
    "IntegrationStatus",
]
