"""Generic, kind-agnostic HTTP surface for integrations.

Plain async Django views — thin wrappers over ``IntegrationService``, translating its
return values/exceptions to HTTP, no business logic of their own. Wire
``django_ai_sdk.integrations.urls`` into the host project's URLconf under
``/api/integrations`` — no web framework beyond Django itself is required.

The OAuth *callback* is a separate endpoint (a browser redirect, not JSON) since the
identity provider is the one redirecting the browser there; see
``django_ai_sdk.integrations.mcp.urls``. Starting the flow has no dedicated endpoint —
that's this module's own ``POST /{name}/connect``, whose ``redirect_url`` the client
navigates to itself.
"""

from __future__ import annotations

from http import HTTPStatus

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.urls import reverse
from django.views.decorators.http import require_http_methods

from django_ai_sdk.integrations.base import IntegrationNotConnectable
from django_ai_sdk.integrations.services import IntegrationService
from django_ai_sdk.permissions import PermissionDenied


def _unauthenticated() -> JsonResponse:
    return JsonResponse({"detail": "Not authenticated"}, status=HTTPStatus.UNAUTHORIZED)


@require_http_methods(["GET"])
async def list_integrations(request: HttpRequest) -> HttpResponse:
    """List every integration the user may use, with status and capability flags."""
    if not request.user.is_authenticated:
        return _unauthenticated()
    rows = await IntegrationService.list_for_user(request.user)
    return JsonResponse([row.model_dump(mode="json") for row in rows], safe=False)


@require_http_methods(["POST"])
async def connect(request: HttpRequest, name: str) -> HttpResponse:
    """Begin connecting an integration (OAuth); returns a redirect URL for the client
    to navigate to itself."""
    if not request.user.is_authenticated:
        return _unauthenticated()
    redirect_uri = request.build_absolute_uri(
        reverse("integrations_mcp:oauth-callback", kwargs={"server_name": name})
    )
    try:
        result = await IntegrationService.connect(
            name, request.user, request=request, redirect_uri=redirect_uri
        )
    except PermissionDenied:
        return JsonResponse({"detail": "Not permitted"}, status=HTTPStatus.FORBIDDEN)
    except IntegrationNotConnectable as e:
        return JsonResponse(
            {"detail": str(e) or "Integration does not support connect"}, status=400
        )
    if result is None:
        return JsonResponse({"detail": "Unknown integration"}, status=404)
    return JsonResponse({"redirect_url": result["redirect_url"]})


@require_http_methods(["POST"])
async def disconnect(request: HttpRequest, name: str) -> HttpResponse:
    """Drop the user's stored connection/credential for an integration."""
    if not request.user.is_authenticated:
        return _unauthenticated()
    try:
        deleted = await IntegrationService.disconnect(name, request.user)
    except PermissionDenied:
        return JsonResponse({"detail": "Not permitted"}, status=HTTPStatus.FORBIDDEN)
    if deleted is None:
        return JsonResponse({"detail": "Unknown integration"}, status=404)
    if not deleted:
        return JsonResponse({"detail": "Not connected"}, status=404)
    return JsonResponse({"detail": f"Disconnected {name}"})


@require_http_methods(["POST"])
async def reconnect(request: HttpRequest, name: str) -> HttpResponse:
    """Force a fresh connection attempt and return the real status."""
    if not request.user.is_authenticated:
        return _unauthenticated()
    try:
        status = await IntegrationService.reconnect(name, request.user)
    except PermissionDenied:
        return JsonResponse({"detail": "Not permitted"}, status=HTTPStatus.FORBIDDEN)
    if status is None:
        return JsonResponse({"detail": "Unknown integration"}, status=404)
    return JsonResponse({"status": status.value})
