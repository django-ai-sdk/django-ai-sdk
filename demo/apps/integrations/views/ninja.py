from __future__ import annotations

from django.http import HttpRequest
from django.urls import reverse
from django_ai_sdk.errors import NotFound
from django_ai_sdk.integrations.base import IntegrationStatus
from django_ai_sdk.integrations.schemas import IntegrationOut
from django_ai_sdk.integrations.services import IntegrationService
from django_ai_sdk.views.schemas import ErrorResponse
from ninja import Router, Schema


class DetailOut(Schema):
    detail: str


class ConnectOut(Schema):
    """Where the client should go to complete a connection (e.g. an OAuth redirect)."""

    redirect_url: str


class StatusOut(Schema):
    status: IntegrationStatus


router = Router()


@router.get(
    "/",
    response={200: list[IntegrationOut]},
)
async def list_integrations(request: HttpRequest) -> list[IntegrationOut]:
    """List every integration this user may use, with its real current status."""
    return await IntegrationService.list_for_user(request.user)


@router.post(
    "/{name}/connect",
    response={200: ConnectOut, 400: ErrorResponse, 403: ErrorResponse, 404: ErrorResponse},
)
async def connect(request: HttpRequest, name: str) -> ConnectOut:
    """Begin connecting an integration (OAuth); returns a redirect URL for the client
    to navigate to itself."""
    redirect_uri = request.build_absolute_uri(
        reverse("integrations_mcp:oauth-callback", kwargs={"server_name": name})
    )
    result = await IntegrationService.connect(
        name, request.user, request=request, redirect_uri=redirect_uri
    )
    if result is None:
        raise NotFound("Unknown integration")
    return ConnectOut(redirect_url=result["redirect_url"])


@router.post(
    "/{name}/disconnect",
    response={200: DetailOut, 403: ErrorResponse, 404: ErrorResponse},
)
async def disconnect(request: HttpRequest, name: str) -> DetailOut:
    """Drop the user's stored connection/credential for an integration."""
    deleted = await IntegrationService.disconnect(name, request.user)
    if deleted is None:
        raise NotFound("Unknown integration")
    if not deleted:
        raise NotFound("Not connected")
    return DetailOut(detail=f"Disconnected {name}")


@router.post(
    "/{name}/reconnect",
    response={200: StatusOut, 403: ErrorResponse, 404: ErrorResponse},
)
async def reconnect(request: HttpRequest, name: str) -> StatusOut:
    """Force a fresh connection attempt and return the real status."""
    status = await IntegrationService.reconnect(name, request.user)
    if status is None:
        raise NotFound("Unknown integration")
    return StatusOut(status=status)
