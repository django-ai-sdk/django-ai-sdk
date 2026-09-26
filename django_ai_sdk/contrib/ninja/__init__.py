"""Ready-made django-ninja endpoints over the SDK services.

Needs ``pip install django-ai-sdk[ninja]``. Mount what you need::

    from django_ai_sdk.contrib import ninja as ai

    api = NinjaAPI(auth=SessionAuth())
    ai.register_error_handlers(api)
    api.add_router("/", ai.get_threads_router())
    api.add_router("/", ai.get_agents_router())
    api.add_router("/", ai.get_workflows_router())
    api.add_router("/memories", ai.get_memories_router())
    api.add_router("/integrations", ai.get_integrations_router())

Each ``get_*_router()`` returns a new Router, so you can drop endpoints with
``exclude={"delete_all_threads"}`` and add your own on the result.
"""

from __future__ import annotations

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
from django_ai_sdk.contrib.ninja.workflows import get_workflows_router

__all__ = [
    "get_agents_router",
    "get_integrations_router",
    "get_memories_router",
    "get_threads_router",
    "get_workflows_router",
    "register_error_handlers",
]
