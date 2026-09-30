"""Integrations: list with live status, connect (OAuth), disconnect, reconnect.

The OAuth *callback* is not here: it must sit at a fixed URL, so the SDK ships it as a
plain Django view (``include("django_ai_sdk.integrations.mcp.urls")``).
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from django.http import HttpRequest
from django.urls import reverse
from ninja import Router

from django_ai_sdk.contrib.ninja.routing import ApiRouter
from django_ai_sdk.contrib.ninja.schemas import (
    ConnectOut,
    DetailOut,
    StatusOut,
)
from django_ai_sdk.errors import NotFound
from django_ai_sdk.integrations.schemas import IntegrationOut
from django_ai_sdk.integrations.services import IntegrationService

routes = ApiRouter()


@routes.get("/", response=list[IntegrationOut])
async def list_integrations(request: HttpRequest) -> Any:
    """Every integration this user may use, with its real current status."""
    return await IntegrationService.list_for_user(request.user)


@routes.post("/{name}/connect", response=ConnectOut)
async def connect_integration(request: HttpRequest, name: str) -> Any:
    """Begin connecting (OAuth); returns the URL the client should navigate to."""
    redirect_uri = request.build_absolute_uri(
        reverse("integrations_mcp:oauth-callback", kwargs={"server_name": name})
    )
    result = await IntegrationService.connect(
        name, request.user, request=request, redirect_uri=redirect_uri
    )
    if result is None:
        raise NotFound("Unknown integration")
    return ConnectOut(redirect_url=result["redirect_url"])


@routes.post("/{name}/disconnect", response=DetailOut)
async def disconnect_integration(request: HttpRequest, name: str) -> Any:
    """Drop the user's stored connection/credential."""
    deleted = await IntegrationService.disconnect(name, request.user)
    if deleted is None:
        raise NotFound("Unknown integration")
    if not deleted:
        raise NotFound("Not connected")
    return DetailOut(detail=f"Disconnected {name}")


@routes.post("/{name}/reconnect", response=StatusOut)
async def reconnect_integration(request: HttpRequest, name: str) -> Any:
    """Force a fresh connection attempt and return the real status."""
    status = await IntegrationService.reconnect(name, request.user)
    if status is None:
        raise NotFound("Unknown integration")
    return StatusOut(status=status)


def get_integrations_router(*, exclude: Collection[str] = (), **router_kwargs: Any) -> Router:
    """A new Router with the integration endpoints; mount with
    ``api.add_router("/integrations", ...)``."""
    return routes.build(exclude=exclude, **router_kwargs)
