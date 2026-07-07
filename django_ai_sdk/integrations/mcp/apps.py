from __future__ import annotations

import logging
import sys
import threading

from django.apps import AppConfig
from django.conf import settings

logger = logging.getLogger(__name__)

# Management commands that shouldn't trigger a network-touching pre-warm as a side
# effect of loading the app registry (migrations, tests, static introspection, etc.).
_SKIP_COMMANDS = {
    "migrate",
    "makemigrations",
    "test",
    "shell",
    "shell_plus",
    "collectstatic",
    "check",
}


class MCPConfig(AppConfig):
    name = "django_ai_sdk.integrations.mcp"
    label = "django_ai_sdk_mcp"
    verbose_name = "Django AI SDK — MCP"

    def ready(self) -> None:
        if not getattr(settings, "AI_SDK_INTEGRATION_PREWARM", True):
            return
        if len(sys.argv) > 1 and sys.argv[1] in _SKIP_COMMANDS:
            return
        threading.Thread(target=_warm_all_sync, daemon=True, name="integration-prewarm").start()


def _warm_all_sync() -> None:
    """Populate the shared cache for every static/token MCP integration.

    Runs in a background thread with its own event loop so a slow/dead server at
    startup can't delay process boot — by the time the first real request arrives,
    static/token integrations are (usually) already warm. OAuth integrations aren't
    pre-warmed here (no per-user variance to warm without a specific user); they pay
    the normal bounded first-fetch cost on that user's first request instead.
    """
    import asyncio

    from django_ai_sdk.integrations.mcp.loader import MCPIntegration
    from django_ai_sdk.integrations.registry import get_all_integrations

    async def _warm() -> None:
        integrations = get_all_integrations()
        mcp_integrations = [i for i in integrations.values() if isinstance(i, MCPIntegration)]
        await asyncio.gather(*(i.warm() for i in mcp_integrations), return_exceptions=True)

    try:
        asyncio.run(_warm())
    except Exception:
        logger.exception("Integration pre-warm failed")
