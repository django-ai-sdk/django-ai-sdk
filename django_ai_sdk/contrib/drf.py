"""Django REST framework wiring for :func:`django_ai_sdk.errors.error_response`.

Set ``REST_FRAMEWORK = {"EXCEPTION_HANDLER": "django_ai_sdk.contrib.drf.exception_handler"}``.
"""

from __future__ import annotations

from typing import Any

from rest_framework.exceptions import APIException
from rest_framework.response import Response
from rest_framework.views import exception_handler as drf_exception_handler

from django_ai_sdk.errors import error_response


def exception_handler(exc: Exception, context: dict[str, Any]) -> Response:
    """DRF's own exceptions keep DRF's handling; everything else gets an error code."""
    if isinstance(exc, APIException) and (response := drf_exception_handler(exc, context)):
        return response
    status, body = error_response(exc)
    return Response(body, status=status)
