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
from django_ai_sdk.contrib import ninja as ai
from ninja import NinjaAPI
from ninja.security import SessionAuth

from apps.agents.views.ninja import router as studio_router

api = NinjaAPI(title="Django AI SDK Demo", version="1.0.0", auth=SessionAuth())
# Every error answers with a stable code, never the exception text.
ai.register_error_handlers(api)

# The SDK's ready-made endpoints...
api.add_router("/", ai.get_threads_router())
api.add_router("/", ai.get_agents_router())
api.add_router("/", ai.get_workflows_router())
api.add_router("/memories", ai.get_memories_router())
api.add_router("/integrations", ai.get_integrations_router())
# ...and the studio's own (health, digest, accounts).
api.add_router("/", studio_router)


urlpatterns = [
    path("admin/", admin.site.urls),
    path("accounts/", include("allauth.urls")),
    path("_allauth/", include("allauth.headless.urls")),
    path("api/", api.urls),
    # The same services as DRF viewsets, for projects built on DRF.
    path("api/v2/", include("django_ai_sdk.contrib.drf.urls")),
    path("api/integrations/", include("django_ai_sdk.integrations.mcp.urls")),
]
