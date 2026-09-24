"""DRF exception handler mapping django-ai-sdk service errors to HTTP responses.

The one place a service error becomes a status, so a view that forgets a catch
answers 403 rather than 500. A view still catches earlier for a custom payload.
"""

from __future__ import annotations

from typing import Any

from django.core.exceptions import ImproperlyConfigured, ObjectDoesNotExist
from django_ai_sdk.permissions import PermissionDenied
from pydantic import ValidationError
from rest_framework.response import Response
from rest_framework.views import exception_handler as drf_exception_handler


def api_exception_handler(exc: Exception, context: dict[str, Any]) -> Response | None:
    if isinstance(exc, PermissionDenied):
        return Response({"detail": str(exc)}, status=403)
    if isinstance(exc, ObjectDoesNotExist):
        return Response({"detail": "Not found"}, status=404)
    if isinstance(exc, ImproperlyConfigured):
        # A workflow definition the engine could not run is the caller's bad request.
        return Response({"detail": str(exc)}, status=400)
    # Ahead of ValueError, which a pydantic ValidationError also is.
    if isinstance(exc, ValidationError):
        return Response({"detail": exc.errors()}, status=400)
    if isinstance(exc, ValueError):
        # Service-layer convention: ValueError means a referenced object was not found.
        return Response({"detail": str(exc)}, status=404)
    return drf_exception_handler(exc, context)
