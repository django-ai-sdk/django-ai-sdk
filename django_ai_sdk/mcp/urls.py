from django.urls import path

from django_ai_sdk.mcp import api, views

app_name = "django_ai_sdk_mcp"

urlpatterns = [
    path("connections/", api.list_connections, name="connections"),
    path("connections/<slug:server_name>/", api.disconnect_server, name="disconnect"),
    path("oauth/<slug:server_name>/start/", views.oauth_start, name="oauth_start"),
    path("oauth/<slug:server_name>/callback/", views.oauth_callback, name="oauth_callback"),
]
