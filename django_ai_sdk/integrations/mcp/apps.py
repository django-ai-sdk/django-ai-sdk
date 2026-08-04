from __future__ import annotations

from django.apps import AppConfig


class MCPConfig(AppConfig):
    """The MCP toolkit app — ships the OAuth token/client models, their migrations,
    and the OAuth redirect views.

    It holds no integrations of its own. Concrete MCP integrations (notion, linear, ...)
    are MCPIntegration subclasses, each its own app; they reuse this app's models.
    This app must be in INSTALLED_APPS alongside any MCP-backed integration.

    ready() does nothing: no network warmup, no registration. Tool lists populate
    lazily on first use via ResilientCache.
    """

    name = "django_ai_sdk.integrations.mcp"
    label = "django_ai_sdk_mcp"
    verbose_name = "Django AI SDK — MCP toolkit"
