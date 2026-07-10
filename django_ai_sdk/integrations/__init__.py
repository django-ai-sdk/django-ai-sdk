"""Integrations: the SDK's uniform capability layer for pluggable third-party
functionality — MCP servers, hand-written API wrappers, and anything added later.

Each integration is a Django app (see ``IntegrationAppConfig``) exposing one
``IntegrationService`` (the shared contract in ``base.py``).
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
