"""OAuth redirect endpoints for MCP integrations.

Include this in the host project's URLconf, e.g.::

    path("api/integrations/", include("django_ai_sdk.integrations.mcp.urls")),

so the reverse names resolve as ``integrations_mcp:oauth-start`` / ``:oauth-callback``.
"""

from __future__ import annotations

from django.urls import path

from django_ai_sdk.integrations.mcp import oauth_views

app_name = "integrations_mcp"

urlpatterns = [
    path("oauth/<str:server_name>/start/", oauth_views.oauth_start, name="oauth-start"),
    path("oauth/<str:server_name>/callback/", oauth_views.oauth_callback, name="oauth-callback"),
]
