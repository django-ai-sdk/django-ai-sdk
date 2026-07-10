from __future__ import annotations

from typing import Any

from django.apps import AppConfig


def _invalidate_on_change(sender: type, instance: Any, **kwargs: Any) -> None:
    """post_save/post_delete receiver for MCPServerConfig — module-level (not a
    closure) so Django's default weak-reference signal connection doesn't garbage
    collect it the moment ``ready()`` returns."""
    from django_ai_sdk.integrations.registry import invalidate_db_service

    invalidate_db_service(instance.name)


class MCPConfig(AppConfig):
    """The MCP toolkit app — ships the OAuth token/client models, the DB-declared
    ``MCPServerConfig`` model, and their migrations.

    It holds no integrations of its own. Concrete MCP integrations (notion, linear,
    …) are separate apps whose ``IntegrationAppConfig`` self-registers a service; they
    reuse this app's models. A DB-declared MCP server (``MCPServerConfig`` row) needs
    no app at all — see ``integrations.registry._db_services``.

    There is no boot-time network warmup anywhere — tool lists populate lazily on
    first use via the cache. ``ready()`` only registers a signal handler (no I/O) that
    invalidates a DB-declared service's cached instance when its row changes.
    """

    name = "django_ai_sdk.integrations.mcp"
    label = "django_ai_sdk_mcp"
    verbose_name = "Django AI SDK — MCP toolkit"

    def ready(self) -> None:
        from django.db.models.signals import post_delete, post_save

        from django_ai_sdk.integrations.mcp.models import MCPServerConfig

        post_save.connect(
            _invalidate_on_change,
            sender=MCPServerConfig,
            dispatch_uid="mcp_server_config_invalidate_save",
        )
        post_delete.connect(
            _invalidate_on_change,
            sender=MCPServerConfig,
            dispatch_uid="mcp_server_config_invalidate_delete",
        )
