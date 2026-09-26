"""Route tables that build fresh Ninja Routers, and the error handlers.

A Ninja ``Router`` can be attached to one API only, so each module keeps a
:class:`Routes` table and hands out a new Router per call (``get_threads_router()``
and friends). ``exclude`` drops endpoints by function name so you can replace them.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from types import FunctionType
from typing import TYPE_CHECKING, Any

from ninja import Router

from django_ai_sdk.errors import error_response
from django_ai_sdk.views.schemas import ErrorResponse

if TYPE_CHECKING:
    from django.http import HttpRequest, HttpResponse
    from ninja import NinjaAPI


# Documented on every operation; raised by services, answered by the handlers below.
ERRORS = {400: ErrorResponse, 403: ErrorResponse, 404: ErrorResponse, 500: ErrorResponse}


class Routes:
    def __init__(self) -> None:
        self._ops: list[tuple[str, str, FunctionType, dict[str, Any]]] = []

    def _add(self, method: str, path: str, **opts: Any) -> Callable[[FunctionType], FunctionType]:
        def register(func: FunctionType) -> FunctionType:
            self._ops.append((method, path, func, opts))
            return func

        return register

    def get(self, path: str, **opts: Any) -> Callable[[FunctionType], FunctionType]:
        return self._add("GET", path, **opts)

    def post(self, path: str, **opts: Any) -> Callable[[FunctionType], FunctionType]:
        return self._add("POST", path, **opts)

    def put(self, path: str, **opts: Any) -> Callable[[FunctionType], FunctionType]:
        return self._add("PUT", path, **opts)

    def patch(self, path: str, **opts: Any) -> Callable[[FunctionType], FunctionType]:
        return self._add("PATCH", path, **opts)

    def delete(self, path: str, **opts: Any) -> Callable[[FunctionType], FunctionType]:
        return self._add("DELETE", path, **opts)

    def build(self, *, exclude: Collection[str] = (), **router_kwargs: Any) -> Router:
        unknown = set(exclude) - {func.__name__ for _, _, func, _ in self._ops}
        if unknown:
            raise ValueError(f"Unknown endpoints in exclude: {sorted(unknown)}")
        router = Router(**router_kwargs)
        for method, path, func, opts in self._ops:
            if func.__name__ in exclude:
                continue
            opts = dict(opts)
            response = opts.pop("response", None)
            if response is None:
                response = ERRORS
            elif not isinstance(response, dict):
                response = {200: response, **ERRORS}
            operation_id = opts.pop("operation_id", func.__name__)
            router.add_api_operation(
                path, [method], func, response=response, operation_id=operation_id, **opts
            )
        return router


def register_error_handlers(api: NinjaAPI) -> None:
    """Answer every unhandled exception with an error code instead of its text.

    Ninja's own handlers (request validation, ``HttpError``, auth) are more
    specific, so they still win.
    """

    @api.exception_handler(Exception)
    def handle(request: HttpRequest, exc: Exception) -> HttpResponse:
        status, body = error_response(exc)
        return api.create_response(request, body, status=status)
