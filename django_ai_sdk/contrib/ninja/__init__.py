"""Ready-made django-ninja endpoints over the SDK services.

Needs ``pip install django-ai-sdk[ninja]``. Mount what you need::

    from django_ai_sdk.contrib import ninja as ai_sdk_routers

    api = NinjaAPI(auth=SessionAuth())
    ai_sdk_routers.register_error_handlers(api)
    api.add_router("/", ai_sdk_routers.get_threads_router())
    api.add_router("/", ai_sdk_routers.get_agents_router())
    api.add_router("/", ai_sdk_routers.get_workflows_router())
    api.add_router("/memories", ai_sdk_routers.get_memories_router())
    api.add_router("/integrations", ai_sdk_routers.get_integrations_router())
    api.add_router("/", ai_sdk_routers.get_oversight_router())

The oversight and workflows routers are imported only when requested, and need
``django_ai_sdk.oversight`` and ``django_ai_sdk.workflows`` in INSTALLED_APPS. The
oversight router adds the authoring routes when that app is installed.

Each ``get_*_router()`` returns a new Router, so you can drop endpoints with
``exclude={"delete_all_threads"}`` and add your own on the result.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

try:
    import ninja  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised without the extra installed
    from django.core.exceptions import ImproperlyConfigured

    raise ImproperlyConfigured(
        "django_ai_sdk.contrib.ninja needs django-ninja: pip install django-ai-sdk[ninja]"
    ) from exc

from django_ai_sdk.contrib.ninja.agents import get_agents_router
from django_ai_sdk.contrib.ninja.integrations import get_integrations_router
from django_ai_sdk.contrib.ninja.memories import get_memories_router
from django_ai_sdk.contrib.ninja.routing import register_error_handlers
from django_ai_sdk.contrib.ninja.threads import get_threads_router

# Imported on first use (PEP 562): each needs its own app installed.
_LAZY_ROUTERS = {"get_oversight_router": "oversight", "get_workflows_router": "workflows"}


def __getattr__(name: str) -> Any:
    if name in _LAZY_ROUTERS:
        return getattr(import_module(f"{__name__}.{_LAZY_ROUTERS[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "get_agents_router",
    "get_integrations_router",
    "get_memories_router",
    "get_oversight_router",
    "get_threads_router",
    "get_workflows_router",
    "register_error_handlers",
]
