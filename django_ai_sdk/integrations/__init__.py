from __future__ import annotations

from django_ai_sdk.integrations.api.base import APIIntegration
from django_ai_sdk.integrations.apps import IntegrationAppConfig
from django_ai_sdk.integrations.base import (
    Integration,
    IntegrationNotConnectable,
    IntegrationStatus,
)

# Safe eagerly: loader.py defers its optional `mcp`/`haystack` imports.
from django_ai_sdk.integrations.mcp.loader import MCPIntegration

__all__ = [
    "APIIntegration",
    "Integration",
    "IntegrationAppConfig",
    "IntegrationNotConnectable",
    "IntegrationStatus",
    "MCPIntegration",
]
