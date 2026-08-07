from __future__ import annotations

from django.apps import AppConfig


class MCPConfig(AppConfig):
    """The MCP toolkit app — ships the OAuth token/client models, their migrations,
    and the OAuth redirect views.

    It holds no integrations of its own. Concrete MCP integrations (notion, linear, ...)
    are MCPIntegration subclasses, each its own app; they reuse this app's models.
    This app must be in INSTALLED_APPS alongside any MCP-backed integration.

    ready() does no network warmup, no registration -- tool lists populate lazily on
    first use via ResilientCache. It does connect a signal so the registry's
    enabled-rows cache (see registry.invalidate_db_rows_cache) drops the moment an
    MCPServerConfig is saved or deleted, instead of waiting out its TTL.
    """

    name = "django_ai_sdk.integrations.mcp"
    label = "django_ai_sdk_mcp"
    verbose_name = "Django AI SDK — MCP toolkit"

    def ready(self) -> None:
        from django.db.models.signals import post_delete, post_save

        from django_ai_sdk.integrations.mcp.models import MCPServerConfig
        from django_ai_sdk.integrations.registry import invalidate_db_rows_cache

        def _invalidate(**kwargs: object) -> None:
            invalidate_db_rows_cache()

        # weak=False: _invalidate is a local closure with nothing else holding a
        # reference to it, so a weak (the default) receiver would be garbage
        # collected the moment ready() returns and never fire.
        post_save.connect(
            _invalidate,
            sender=MCPServerConfig,
            dispatch_uid="mcp_server_config_saved",
            weak=False,
        )
        post_delete.connect(
            _invalidate,
            sender=MCPServerConfig,
            dispatch_uid="mcp_server_config_deleted",
            weak=False,
        )
