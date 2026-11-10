"""Oversight routes: turns, review queues and metrics, plus the installed opt-in apps' routes.

The router signs readers in with ``django_auth`` (session, CSRF on POST) unless you pass
``auth=``; it does not inherit the API's auth. Permissions and the access record are in
the oversight manual's Review API::

    api.add_router("/", get_oversight_router())
"""

from __future__ import annotations

from collections.abc import Collection
from importlib import import_module
from typing import Any

from django.http import HttpRequest
from ninja import Query, Router
from ninja.security import django_auth

from django_ai_sdk.contrib.ninja.routing import ApiRouter, Limit, found
from django_ai_sdk.oversight.schemas import (
    MetricsOut,
    Page,
    ResolveIn,
    ReviewOut,
    ReviewStatus,
    TurnDetail,
    TurnFilter,
    TurnOut,
)
from django_ai_sdk.oversight.services import OversightService

routes = ApiRouter()


@routes.get("/oversight/turns/", response=Page[TurnOut])
async def list_oversight_turns(
    request: HttpRequest, filters: Query[TurnFilter], limit: Limit = 50, cursor: str = ""
) -> Any:
    return await OversightService.list_turns(filters, user=request.user, limit=limit, cursor=cursor)


@routes.get("/oversight/turns/{turn_id}/", response=TurnDetail)
async def get_oversight_turn(request: HttpRequest, turn_id: str) -> Any:
    return found(await OversightService.get_turn(turn_id, user=request.user), "Turn")


@routes.get("/oversight/reviews/", response=Page[ReviewOut])
async def list_oversight_reviews(
    request: HttpRequest,
    queue: str = "",
    status: ReviewStatus = "open",
    limit: Limit = 50,
    cursor: str = "",
) -> Any:
    return await OversightService.list_reviews(
        user=request.user, queue=queue, status=status, limit=limit, cursor=cursor
    )


@routes.get("/oversight/queues/", response=dict[str, int])
async def list_oversight_queues(request: HttpRequest) -> Any:
    return await OversightService.queues(user=request.user)


# 409 when the item is already closed.
@routes.post("/oversight/reviews/{item_id}/", response=ReviewOut)
async def resolve_oversight_review(request: HttpRequest, item_id: str, payload: ResolveIn) -> Any:
    item = await OversightService.resolve_review(item_id, payload, user=request.user)
    return found(item, "Review item")


@routes.get("/oversight/metrics/", response=MetricsOut)
async def get_oversight_metrics(
    request: HttpRequest, days: int = 14, agent_id: str = "", policy: str = ""
) -> Any:
    return await OversightService.summarize(
        user=request.user, days=days, agent_id=agent_id, policy=policy
    )


def get_oversight_router(*, exclude: Collection[str] = (), **router_kwargs: Any) -> Router:
    """Build the oversight router. Mount it with ``api.add_router("/", ...)``.

    Uses ``django_auth`` unless ``auth=`` is passed.
    """
    from django_ai_sdk.oversight.apps import ROUTE_MODULES

    # The installed opt-in apps' routes, which they added on start.
    table = ApiRouter(routes, *(import_module(module).routes for module in ROUTE_MODULES))
    return table.build(exclude=exclude, default_auth=django_auth, **router_kwargs)


__all__ = ["get_oversight_router"]
