from __future__ import annotations

from django.apps import AppConfig


class MCPConfig(AppConfig):
    """The MCP toolkit app — ships the OAuth token/client models, their migrations,
    and the OAuth redirect views.

    It holds no integrations of its own. Concrete MCP integrations (notion, linear, …)
    are ``MCPIntegrationService`` subclasses listed in ``AI_SDK_INTEGRATIONS``; they
    reuse this app's models. Only this one app needs to be in ``INSTALLED_APPS`` —
    there is no app per integration.

    ``ready()`` deliberately does nothing: no network warmup, no registration. Tool
    lists populate lazily on first use via ``ResilientCache``.
    """

    name = "django_ai_sdk.integrations.mcp"
    label = "django_ai_sdk_mcp"
    verbose_name = "Django AI SDK — MCP toolkit"
