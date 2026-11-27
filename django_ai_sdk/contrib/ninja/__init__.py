from __future__ import annotations

from importlib import import_module
from typing import Any

try:
    import ninja  # noqa: F401
except ImportError as exc:  # pragma: no cover
    from django.core.exceptions import ImproperlyConfigured

    raise ImproperlyConfigured(
        "django_ai_sdk.contrib.ninja needs django-ninja: pip install django-ai-sdk[ninja]"
    ) from exc

from django_ai_sdk.contrib.ninja.agents import get_agents_router
from django_ai_sdk.contrib.ninja.integrations import get_integrations_router
from django_ai_sdk.contrib.ninja.memories import get_memories_router
from django_ai_sdk.contrib.ninja.routing import register_error_handlers
from django_ai_sdk.contrib.ninja.threads import get_threads_router

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
