from __future__ import annotations

try:
    import rest_framework  # noqa: F401
except ImportError as exc:  # pragma: no cover
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
