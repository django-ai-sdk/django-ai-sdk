"""API-backed integrations: tools written directly in code, not discovered via MCP.

See APIIntegration for the base class.
"""

from __future__ import annotations

from django_ai_sdk.integrations.api.base import APIIntegration

__all__ = ["APIIntegration"]
