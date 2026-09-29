"""django-ninja wiring for :func:`django_ai_sdk.errors.error_response`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from django_ai_sdk.errors import error_response

if TYPE_CHECKING:
    from django.http import HttpRequest, HttpResponse
    from ninja import NinjaAPI


def register_error_handlers(api: NinjaAPI) -> None:
    """Answer every unhandled exception with an error code instead of its text.

    Ninja's own handlers (request validation, ``HttpError``, auth) are more
    specific, so they still win.
    """

    @api.exception_handler(Exception)
    def handle(request: HttpRequest, exc: Exception) -> HttpResponse:
        status, body = error_response(exc)
        return api.create_response(request, body, status=status)
