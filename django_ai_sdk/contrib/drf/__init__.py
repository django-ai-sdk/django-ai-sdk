"""Ready-made Django REST framework viewsets over the SDK services.

Needs ``pip install django-ai-sdk[drf]``. Use them all::

    path("api/", include("django_ai_sdk.contrib.drf.urls"))

or subclass one and register it on your own router::

    class MyThreadViewSet(ThreadViewSet):
        throttle_classes = [UserRateThrottle]

    router.register("threads", MyThreadViewSet, basename="thread")

Errors answer with an error code (see ``django_ai_sdk.errors``), whatever your
``EXCEPTION_HANDLER`` is. Set ``REST_FRAMEWORK["EXCEPTION_HANDLER"] =
"django_ai_sdk.contrib.drf.exception_handler"`` to give your own views the same.
"""

from __future__ import annotations

try:
    import rest_framework  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised without the extra installed
    from django.core.exceptions import ImproperlyConfigured

    raise ImproperlyConfigured(
        "django_ai_sdk.contrib.drf needs djangorestframework: pip install django-ai-sdk[drf]"
    ) from exc

from django_ai_sdk.contrib.drf.base import ApiPagination, ApiViewSet, exception_handler
from django_ai_sdk.contrib.drf.views import (
    AgentViewSet,
    IntegrationViewSet,
    MemoryViewSet,
    MessageViewSet,
    RuntimeAgentViewSet,
    ThreadViewSet,
    WorkflowViewSet,
)

__all__ = [
    "AgentViewSet",
    "ApiPagination",
    "ApiViewSet",
    "IntegrationViewSet",
    "MemoryViewSet",
    "MessageViewSet",
    "RuntimeAgentViewSet",
    "ThreadViewSet",
    "WorkflowViewSet",
    "exception_handler",
]
