from django.urls import include, path

urlpatterns = [
    path("api/integrations/", include("django_ai_sdk.integrations.mcp.urls")),
]
