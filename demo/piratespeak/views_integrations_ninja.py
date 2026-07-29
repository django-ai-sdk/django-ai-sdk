from __future__ import annotations

from django.http import HttpRequest
from django.urls import reverse
from django_ai_sdk.integrations.base import IntegrationNotConnectable, IntegrationStatus
from django_ai_sdk.integrations.schemas import IntegrationOut
from django_ai_sdk.integrations.services import IntegrationService
from django_ai_sdk.permissions import PermissionDenied
from ninja import Router, Schema

from .views_ninja import Error


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
    """Return upload constraints for the frontend."""
    return await IntegrationService.list_for_user(request.user)


@router.post(
    "/{name}/connect",
    response={200: ConnectOut, 400: Error, 403: Error, 404: Error},
)
async def connect(request: HttpRequest, name: str) -> ConnectOut | tuple[int, Error]:
    """Begin connecting an integration (OAuth); returns a redirect URL for the client
    to navigate to itself."""
    redirect_uri = request.build_absolute_uri(
        reverse("integrations_mcp:oauth-callback", kwargs={"server_name": name})
    )
    try:
        result = await IntegrationService.connect(
            name, request.user, request=request, redirect_uri=redirect_uri
        )
    except PermissionDenied:
        return 403, Error(message="Not permitted")
    except IntegrationNotConnectable as e:
        return 400, Error(message=str(e) or "Integration does not support connect")
    if result is None:
        return 404, Error(message="Unknown integration")
    return ConnectOut(redirect_url=result["redirect_url"])


@router.post(
    "/{name}/disconnect",
    response={200: DetailOut, 403: Error, 404: Error},
)
async def disconnect(request: HttpRequest, name: str) -> DetailOut | tuple[int, Error]:
    """Drop the user's stored connection/credential for an integration."""
    try:
        deleted = await IntegrationService.disconnect(name, request.user)
    except PermissionDenied:
        return 403, Error(message="Not permitted")
    if deleted is None:
        return 404, Error(message="Unknown integration")
    if not deleted:
        return 404, Error(message="Not connected")
    return DetailOut(detail=f"Disconnected {name}")


@router.post(
    "/{name}/reconnect",
    response={200: StatusOut, 403: Error, 404: Error},
)
async def reconnect(request: HttpRequest, name: str) -> StatusOut | tuple[int, Error]:
    """Force a fresh connection attempt and return the real status."""
    try:
        status = await IntegrationService.reconnect(name, request.user)
    except PermissionDenied:
        return 403, Error(message="Not permitted")
    if status is None:
        return 404, Error(message="Unknown integration")
    return StatusOut(status=status)
