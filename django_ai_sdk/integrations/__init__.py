"""Integrations: the SDK's uniform capability layer for pluggable third-party
functionality — MCP servers, hand-written API wrappers, and anything added later.

See django_ai_sdk.integrations.base.Integration for the shared contract.
"""

from __future__ import annotations

from django_ai_sdk.integrations.api.base import APIIntegration
from django_ai_sdk.integrations.base import Integration, IntegrationStatus

__all__ = ["APIIntegration", "Integration", "IntegrationStatus"]
