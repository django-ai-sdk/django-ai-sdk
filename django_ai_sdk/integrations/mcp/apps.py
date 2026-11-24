from __future__ import annotations

from django.apps import AppConfig


class MCPConfig(AppConfig):
    """MCP OAuth models and views; required alongside any MCP-backed integration."""

    name = "django_ai_sdk.integrations.mcp"
    label = "django_ai_sdk_mcp"
    verbose_name = "Django AI SDK — MCP toolkit"

    def ready(self) -> None:
        from django.db.models.signals import post_delete, post_save

        from django_ai_sdk.integrations.mcp.models import MCPServerConfig
        from django_ai_sdk.integrations.registry import invalidate_db_rows_cache

        def _invalidate(**kwargs: object) -> None:
            invalidate_db_rows_cache()

        # weak=False: nothing else holds this closure, so a weak receiver would be collected.
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
