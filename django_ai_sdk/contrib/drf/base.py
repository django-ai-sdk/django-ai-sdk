"""Shared plumbing for the DRF viewsets: error codes, pagination, per-action serializers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from rest_framework import serializers, viewsets
from rest_framework.exceptions import APIException
from rest_framework.pagination import LimitOffsetPagination
from rest_framework.response import Response
from rest_framework.views import exception_handler as drf_exception_handler

from django_ai_sdk.errors import error_response

if TYPE_CHECKING:
    from collections.abc import Callable


def exception_handler(exc: Exception, context: dict[str, Any]) -> Response:
    """DRF's own exceptions keep DRF's handling; everything else gets an error code.

    Used by the SDK viewsets automatically; set it as
    ``REST_FRAMEWORK["EXCEPTION_HANDLER"]`` to give your own views the same codes.
    """
    if isinstance(exc, APIException) and (response := drf_exception_handler(exc, context)):
        return response
    status, body = error_response(exc)
    return Response(body, status=status)


class ApiPagination(LimitOffsetPagination):
    """``?limit=&offset=`` for the list actions.

    The services page in the database, so only the parameters are read; responses stay
    plain lists (the services return no totals for a ``count``).
    """

    default_limit = 100
    # Same bound as the Ninja layer. Services accept `limit=None` (everything);
    # over HTTP a client pages through with `offset`.
    max_limit = 100


class ApiViewSet(viewsets.GenericViewSet):
    """Base for the SDK's viewsets.

    Auth, permission and throttle classes come from your ``REST_FRAMEWORK`` settings
    (or set them on a subclass); per-object access is decided by the services. Request
    serializers are looked up per action in ``serializer_classes``, so a subclass can
    swap one without touching the action.
    """

    pagination_class = ApiPagination
    serializer_classes: ClassVar[dict[str, type[serializers.Serializer]]] = {}

    def get_serializer_class(self) -> type[serializers.Serializer]:
        # Actions without a request body get an empty serializer (browsable API, schemas).
        return self.serializer_classes.get(self.action or "", serializers.Serializer)

    def get_exception_handler(self) -> Callable[..., Response]:
        # Set here, so the SDK views answer with error codes whatever your settings say.
        return exception_handler

    def page(self, request: Any) -> tuple[int, int]:
        """``(limit, offset)`` from the query string, bounded by ``pagination_class``."""
        paginator = self.paginator
        assert isinstance(paginator, LimitOffsetPagination)
        return paginator.get_limit(request) or paginator.default_limit, paginator.get_offset(
            request
        )
