from __future__ import annotations

from django.urls import path

from django_ai_sdk.integrations.mcp import oauth_views

app_name = "integrations_mcp"

urlpatterns = [
    path("oauth/<str:server_name>/callback/", oauth_views.oauth_callback, name="oauth-callback"),
]
