"""Studio URLs: the SDK's Ninja API at /api/, its DRF viewsets at /api/v2/."""

from __future__ import annotations

from django.contrib import admin
from django.urls import include, path
from django_ai_sdk.contrib import ninja as ai_sdk_routers
from ninja import NinjaAPI
from ninja.security import SessionAuth

from apps.agents.views.ninja import router as studio_router

api = NinjaAPI(title="Django AI SDK Demo", version="1.0.0", auth=SessionAuth())
# Every error answers with a stable code, never the exception text.
ai_sdk_routers.register_error_handlers(api)

# The SDK's ready-made endpoints...
api.add_router("/", ai_sdk_routers.get_threads_router())
api.add_router("/", ai_sdk_routers.get_agents_router())
api.add_router("/", ai_sdk_routers.get_workflows_router())
api.add_router("/memories", ai_sdk_routers.get_memories_router())
api.add_router("/integrations", ai_sdk_routers.get_integrations_router())
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
