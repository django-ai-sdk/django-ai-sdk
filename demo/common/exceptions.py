"""DRF exception handler mapping django-ai-sdk service errors to HTTP responses.

Global safety net so service-layer errors never surface as 500s. Views may
still catch these earlier for custom payloads.
"""

from __future__ import annotations

from typing import Any

from django.core.exceptions import ObjectDoesNotExist
from django_ai_sdk.permissions import PermissionDenied
from rest_framework.response import Response
from rest_framework.views import exception_handler as drf_exception_handler


def api_exception_handler(exc: Exception, context: dict[str, Any]) -> Response | None:
    if isinstance(exc, PermissionDenied):
        return Response({"detail": str(exc)}, status=403)
    if isinstance(exc, ObjectDoesNotExist):
        return Response({"detail": "Not found"}, status=404)
    if isinstance(exc, ValueError):
        # Service-layer convention: ValueError means a referenced object was not found.
        return Response({"detail": str(exc)}, status=404)
    return drf_exception_handler(exc, context)
