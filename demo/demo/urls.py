"""
URL configuration for demo project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from apps.assistants.views.ninja import router as assistants_router
from apps.integrations.views.ninja import router as integrations_router
from apps.memories.views.ninja import router as memories_router
from django.contrib import admin
from django.core.exceptions import ObjectDoesNotExist
from django.urls import include, path
from django_ai_sdk.permissions import PermissionDenied
from ninja import NinjaAPI
from ninja.security import SessionAuth

if TYPE_CHECKING:
    from django.http import HttpRequest, HttpResponse


# Create the main API instance
api = NinjaAPI(title="Django AI SDK Demo", version="1.0.0", auth=SessionAuth())

api.add_router("/", assistants_router)
api.add_router("/memories", memories_router)
# The SDK ships no integrations router — HTTP surfaces are the host project's, so it
# doesn't pick your web framework. views_integrations_ninja builds one over
# IntegrationService; the OAuth *callback* is the one leg the SDK does ship, since it
# must sit at a fixed URL (included in urlpatterns below).
api.add_router("/integrations", integrations_router)


# Global safety net so service-layer errors never surface as 500s.
# Endpoints may still catch these earlier for custom payloads.
@api.exception_handler(PermissionDenied)
def _on_permission_denied(request: HttpRequest, exc: PermissionDenied) -> HttpResponse:
    return api.create_response(request, {"detail": str(exc)}, status=403)


@api.exception_handler(ObjectDoesNotExist)
def _on_does_not_exist(request: HttpRequest, exc: ObjectDoesNotExist) -> HttpResponse:
    return api.create_response(request, {"detail": "Not found"}, status=404)


@api.exception_handler(ValueError)
def _on_value_error(request: HttpRequest, exc: ValueError) -> HttpResponse:
    # Service-layer convention: ValueError means a referenced object was not found.
    return api.create_response(request, {"detail": str(exc)}, status=404)


urlpatterns = [
    path("admin/", admin.site.urls),
    path("accounts/", include("allauth.urls")),
    path("_allauth/", include("allauth.headless.urls")),
    path("api/", api.urls),
    path("api/v2/", include("apps.assistants.views.drf")),
    path("api/v2/", include("apps.memories.views.drf")),
    path("api/integrations/", include("django_ai_sdk.integrations.mcp.urls")),
]
