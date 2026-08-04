"""Integrations: the SDK's uniform capability layer for pluggable third-party
functionality, such as MCP servers, hand-written API wrappers, and anything added
later.

Every integration is one Integration (the contract in base.py), living in its own
Django app. Its AppConfig (subclassing IntegrationAppConfig) registers it on
ready(). The assistant and the host project's integrations endpoints all talk to
the one Integration contract, so none of them know or care which kind an
integration is.

Adding one -- three files, no models, no migrations::

    myapp/integrations/zendesk/__init__.py   (empty)
    myapp/integrations/zendesk/apps.py       ZendeskConfig(IntegrationAppConfig)
    myapp/integrations/zendesk/integration.py ZendeskIntegration(MCPIntegration)

    INSTALLED_APPS = [
        "django_ai_sdk",
        "django_ai_sdk.integrations.mcp",   # only for MCP-backed integrations
        "myapp.integrations.zendesk",
    ]

    AI_SDK_INTEGRATIONS = {
        "zendesk": {"TOKEN": env("ZENDESK_API_TOKEN")},
    }

Then list it on an assistant: ``integrations = ["zendesk"]``.

INSTALLED_APPS decides which integrations exist; AI_SDK_INTEGRATIONS configures them,
keyed by registry name, in the shape Django uses for every other pluggable backend
(DATABASES, CACHES, STORAGES). See config.py. The shipped apps are working
templates: weather/ is the API-backed example, github/, linear/ and notion/ are the
MCP-backed ones (token, token and OAuth respectively).

Package layout, two tiers:

- Framework (this directory's top-level modules): base.py (the contract plus
  ResilientCache), registry.py, config.py (AI_SDK_INTEGRATIONS reads),
  services.py (IntegrationService, the facade an HTTP layer calls), and apps.py.
  Nothing here knows about MCP or any specific provider.
- Kinds (api/, mcp/): base classes and shared infra for a category of integration.
  api.base.APIIntegration is the base for hand-written API wrappers. mcp/ is the MCP
  toolkit (OAuth token models, discovery/PKCE, the Haystack MCPToolset loader) for
  any MCP server. Both kinds produce ordinary Haystack tool objects, consumed
  identically by the assistant.

The SDK ships no integrations router: HTTP surfaces are the host project's, since
the SDK doesn't pick your framework. Build one over IntegrationService (list, connect,
disconnect, reconnect) -- demo/piratespeak/views_integrations_ninja.py is a complete
reference -- and include django_ai_sdk.integrations.mcp.urls for the OAuth callback,
which must live at a fixed URL.

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

# Safe to import eagerly: loader.py defers its `mcp`/`haystack` imports into the
# connect path, so this does not drag the optional [mcp] extra into every import of
# this package. Exported so both kinds are reachable the same way -- having
# APIIntegration here but not MCPIntegration sent people hunting through mcp/.
from django_ai_sdk.integrations.mcp.loader import MCPIntegration

__all__ = [
    "APIIntegration",
    "Integration",
    "IntegrationAppConfig",
    "IntegrationNotConnectable",
    "IntegrationStatus",
    "MCPIntegration",
]
