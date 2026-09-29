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

from django.contrib import admin
from django.urls import include, path
from django_ai_sdk.contrib.ninja import register_error_handlers
from ninja import NinjaAPI
from ninja.security import SessionAuth

from apps.agents.views.ninja import router as agents_router
from apps.integrations.views.ninja import router as integrations_router
from apps.memories.views.ninja import router as memories_router

# Create the main API instance
api = NinjaAPI(title="Django AI SDK Demo", version="1.0.0", auth=SessionAuth())

api.add_router("/", agents_router)
api.add_router("/memories", memories_router)
# The SDK ships no integrations router — HTTP surfaces are the host project's, so it
# doesn't pick your web framework. views_integrations_ninja builds one over
# IntegrationService; the OAuth *callback* is the one leg the SDK does ship, since it
# must sit at a fixed URL (included in urlpatterns below).
api.add_router("/integrations", integrations_router)

register_error_handlers(api)


urlpatterns = [
    path("admin/", admin.site.urls),
    path("accounts/", include("allauth.urls")),
    path("_allauth/", include("allauth.headless.urls")),
    path("api/", api.urls),
    path("api/v2/", include("apps.agents.views.drf")),
    path("api/v2/", include("apps.memories.views.drf")),
    path("api/integrations/", include("django_ai_sdk.integrations.mcp.urls")),
]
