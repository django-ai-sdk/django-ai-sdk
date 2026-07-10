"""Generic, kind-agnostic HTTP surface for integrations.

One Ninja router for every integration regardless of kind. It dispatches to the
polymorphic ``IntegrationService`` contract (``connect``/``disconnect``/``test``/
``store_credential``/``get_status``) and never branches on ``kind`` — a new
integration kind needs no change here. Mount it on the host project's ``NinjaAPI`` at
``/api/integrations``.

The OAuth *redirect* endpoints are separate (browser redirects, not JSON); see
``django_ai_sdk.integrations.mcp.urls``.
"""

from __future__ import annotations

import asyncio
import logging
from http import HTTPStatus

from django.http import HttpRequest
from django.urls import NoReverseMatch, reverse
from ninja import Router, Schema
from ninja.security import SessionAuth

from django_ai_sdk.integrations.base import (
    IntegrationNotConnectable,
    IntegrationService,
    IntegrationStatus,
)
from django_ai_sdk.integrations.registry import get_all_integrations, get_integrations
from django_ai_sdk.integrations.schemas import IntegrationOut
from django_ai_sdk.permissions import Operation

logger = logging.getLogger(__name__)

router = Router()


class ConnectOut(Schema):
    """Where the client should go to complete a connection (e.g. an OAuth redirect)."""

    redirect_url: str


class CredentialIn(Schema):
    secret: str


class StatusOut(Schema):
    status: str


class DetailOut(Schema):
    detail: str


def _connect_url(request: HttpRequest, name: str) -> str | None:
    """Reverse the OAuth start URL for a connectable integration, if one exists."""
    try:
        return reverse("integrations_mcp:oauth-start", kwargs={"server_name": name})
    except NoReverseMatch:
        return None


async def _safe_status(name: str, svc: IntegrationService, user: object) -> IntegrationOut | None:
    """One integration's row for the list endpoint. Isolated: a slow/broken
    integration's status call can't stall or break the rest of the list."""
    if not await svc.has_perms(user, Operation.USE_INTEGRATION):
        return None
    try:
        status = await svc.get_status(user)
    except Exception:
        logger.exception("Failed to get status for integration %r", name)
        status = IntegrationStatus.DEGRADED
    return IntegrationOut(
        name=name,
        label=svc.label,
        kind=svc.kind,
        status=status,
        supports_connect=svc.supports_connect,
        supports_test=svc.supports_test,
        connect_kind=svc.connect_kind,
        detail=svc.detail,
        connected=status == IntegrationStatus.ACTIVE,
        connect_url=None,  # filled in below — needs the request, unlike the rest
    )


@router.get("/", response=list[IntegrationOut], auth=SessionAuth())
async def list_integrations(request: HttpRequest) -> list[IntegrationOut]:
    """List every integration the user may use, with status and capability flags."""
    user = request.user
    if not user.is_authenticated:
        return []

    integrations = await get_all_integrations()
    # Concurrent, not sequential — one cold/dead integration must not make the whole
    # settings page pay N x its timeout.
    rows = await asyncio.gather(
        *(_safe_status(name, svc, user) for name, svc in integrations.items())
    )
    result = [row for row in rows if row is not None]
    for row in result:
        svc = integrations[row.name]
        row.connect_url = _connect_url(request, row.name) if svc.supports_connect else None
    return result


@router.post(
    "/{name}/connect",
    response={200: ConnectOut, 400: DetailOut, 403: DetailOut, 404: DetailOut},
    auth=SessionAuth(),
)
async def connect(request: HttpRequest, name: str) -> tuple[int, object]:
    """Begin connecting an integration (OAuth); returns a redirect URL for the client."""
    user = request.user
    svc = (await get_integrations([name])).get(name)
    if svc is None:
        return 404, DetailOut(detail="Unknown integration")
    if not await svc.has_perms(user, Operation.MANAGE_INTEGRATION):
        return HTTPStatus.FORBIDDEN, DetailOut(detail="Not permitted")

    redirect_uri = request.build_absolute_uri(
        reverse("integrations_mcp:oauth-callback", kwargs={"server_name": name})
    )
    try:
        result = await svc.connect(user, request=request, redirect_uri=redirect_uri)
    except IntegrationNotConnectable as e:
        return 400, DetailOut(detail=str(e) or "Integration does not support connect")
    return 200, ConnectOut(redirect_url=result["redirect_url"])


@router.post(
    "/{name}/credential",
    response={200: DetailOut, 400: DetailOut, 403: DetailOut, 404: DetailOut},
    auth=SessionAuth(),
)
async def store_credential(
    request: HttpRequest, name: str, payload: CredentialIn
) -> tuple[int, DetailOut]:
    """Submit a user-supplied secret (``connect_kind == "credential"``, e.g. a
    per-user API token) for an integration that doesn't use an OAuth redirect."""
    user = request.user
    svc = (await get_integrations([name])).get(name)
    if svc is None:
        return 404, DetailOut(detail="Unknown integration")
    if not await svc.has_perms(user, Operation.MANAGE_INTEGRATION):
        return HTTPStatus.FORBIDDEN, DetailOut(detail="Not permitted")
    try:
        await svc.store_credential(user, payload.secret)
    except IntegrationNotConnectable as e:
        return 400, DetailOut(detail=str(e) or "Integration does not accept a stored credential")
    except ValueError as e:
        return 400, DetailOut(detail=str(e))
    return 200, DetailOut(detail=f"Credential stored for {name}")


@router.post(
    "/{name}/disconnect",
    response={200: DetailOut, 403: DetailOut, 404: DetailOut},
    auth=SessionAuth(),
)
async def disconnect(request: HttpRequest, name: str) -> tuple[int, DetailOut]:
    """Drop the user's stored connection/credential for an integration."""
    user = request.user
    svc = (await get_integrations([name])).get(name)
    if svc is None:
        return 404, DetailOut(detail="Unknown integration")
    if not await svc.has_perms(user, Operation.MANAGE_INTEGRATION):
        return HTTPStatus.FORBIDDEN, DetailOut(detail="Not permitted")
    deleted = await svc.disconnect(user)
    if not deleted:
        return 404, DetailOut(detail="Not connected")
    return 200, DetailOut(detail=f"Disconnected {name}")


@router.post(
    "/{name}/reconnect",
    response={200: StatusOut, 403: DetailOut, 404: DetailOut},
    auth=SessionAuth(),
)
async def reconnect(request: HttpRequest, name: str) -> tuple[int, object]:
    """Force a fresh connection attempt and return the real status."""
    user = request.user
    svc = (await get_integrations([name])).get(name)
    if svc is None:
        return 404, DetailOut(detail="Unknown integration")
    if not await svc.has_perms(user, Operation.USE_INTEGRATION):
        return HTTPStatus.FORBIDDEN, DetailOut(detail="Not permitted")
    status = await svc.test(user)
    return 200, StatusOut(status=status.value)
