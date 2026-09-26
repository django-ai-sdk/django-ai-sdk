"""Shared plumbing for the DRF viewsets: error codes and pydantic in/out."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from asgiref.sync import async_to_sync
from pydantic import BaseModel
from rest_framework import viewsets
from rest_framework.exceptions import APIException
from rest_framework.response import Response
from rest_framework.views import exception_handler as drf_exception_handler

from django_ai_sdk.errors import error_response

if TYPE_CHECKING:
    from collections.abc import Callable

M = TypeVar("M", bound=BaseModel)


def exception_handler(exc: Exception, context: dict[str, Any]) -> Response:
    """DRF's own exceptions keep DRF's handling; everything else gets an error code.

    Used by the SDK viewsets automatically; set it as
    ``REST_FRAMEWORK["EXCEPTION_HANDLER"]`` to give your own views the same codes.
    """
    if isinstance(exc, APIException) and (response := drf_exception_handler(exc, context)):
        return response
    status, body = error_response(exc)
    return Response(body, status=status)


class Page(BaseModel):
    limit: int = 100
    offset: int = 0


def to_data(obj: Any) -> Any:
    """Make service return values (pydantic models, dataclasses, lists) renderable."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, list | tuple):
        return [to_data(item) for item in obj]
    return obj


class SDKViewSet(viewsets.ViewSet):
    """Base for the SDK's viewsets. Auth, permission and throttle classes come from
    your ``REST_FRAMEWORK`` settings (or set them on a subclass); per-object access
    is decided by the SDK services themselves."""

    def get_exception_handler(self) -> Callable[..., Response]:
        # Set here, so the SDK views answer with error codes whatever your settings say.
        return exception_handler

    @staticmethod
    def call(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run an async service method from a sync DRF view."""
        return async_to_sync(func)(*args, **kwargs)

    @staticmethod
    def payload(model: type[M], data: Any) -> M:
        return model.model_validate(data)

    @staticmethod
    def page(request: Any, **defaults: int) -> Page:
        return Page.model_validate({**defaults, **request.query_params.dict()})
