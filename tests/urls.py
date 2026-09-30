from django.urls import include, path
from ninja import NinjaAPI
from ninja.security import SessionAuth

from django_ai_sdk.contrib import ninja as ai_sdk_routers
from django_ai_sdk.views.chat import ChatView

api = NinjaAPI(auth=SessionAuth(), urls_namespace="ai-test")
ai_sdk_routers.register_error_handlers(api)
api.add_router("/", ai_sdk_routers.get_threads_router())
api.add_router("/", ai_sdk_routers.get_agents_router())
api.add_router("/", ai_sdk_routers.get_workflows_router())
api.add_router("/memories", ai_sdk_routers.get_memories_router())
api.add_router("/integrations", ai_sdk_routers.get_integrations_router())

urlpatterns = [
    path("api/", api.urls),
    path("drf/", include("django_ai_sdk.contrib.drf.urls")),
    path("api/integrations/", include("django_ai_sdk.integrations.mcp.urls")),
    path("chat/<str:thread_id>/", ChatView.as_view()),
    path("stateless-chat/", ChatView.as_view(agent="test-agent")),
    path("agentless-chat/", ChatView.as_view()),
]
