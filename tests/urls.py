from django.urls import include, path

from django_ai_sdk.views.chat import ChatView

urlpatterns = [
    path("api/integrations/", include("django_ai_sdk.integrations.mcp.urls")),
    path("chat/<str:thread_id>/", ChatView.as_view()),
]
