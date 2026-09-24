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
