from __future__ import annotations

from collections.abc import Callable, Collection
from types import FunctionType
from typing import TYPE_CHECKING, Annotated, Any

from ninja import Router
from pydantic import Field

from django_ai_sdk.common import MAX_PAGE_SIZE
from django_ai_sdk.errors import NotFound, error_response
from django_ai_sdk.views.schemas import ErrorResponse

if TYPE_CHECKING:
    from django.http import HttpRequest, HttpResponse
    from ninja import NinjaAPI


Limit = Annotated[int, Field(ge=1, le=MAX_PAGE_SIZE)]
Offset = Annotated[int, Field(ge=0)]

ERRORS = {code: ErrorResponse for code in (400, 403, 404, 409, 500)}


def found(value: Any, what: str) -> Any:
    if value is None:
        raise NotFound(f"{what} not found")
    return value


class ApiRouter:
    def __init__(self, *tables: ApiRouter) -> None:
        self._ops: list[tuple[str, str, FunctionType, dict[str, Any]]] = [
            op for table in tables for op in table._ops
        ]

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

    def build(
        self, *, exclude: Collection[str] = (), default_auth: Any = None, **router_kwargs: Any
    ) -> Router:
        if default_auth is not None:
            router_kwargs.setdefault("auth", default_auth)
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
                response = {}
            elif not isinstance(response, dict):
                response = {200: response}
            response = {**response, **{c: s for c, s in ERRORS.items() if c not in response}}
            operation_id = opts.pop("operation_id", func.__name__)
            router.add_api_operation(
                path, [method], func, response=response, operation_id=operation_id, **opts
            )
        return router


def register_error_handlers(api: NinjaAPI) -> None:
    """Answer every unhandled exception with an error code instead of its text."""

    @api.exception_handler(Exception)
    def handle(request: HttpRequest, exc: Exception) -> HttpResponse:
        status, body = error_response(exc)
        return api.create_response(request, body, status=status)
