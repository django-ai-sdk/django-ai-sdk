"""OAuth callback endpoint for MCP integrations.

Include this in the host project's URLconf, e.g.::

    path("api/integrations/", include("django_ai_sdk.integrations.mcp.urls")),

so the reverse name resolves as ``integrations_mcp:oauth-callback``. There is no
``oauth-start`` URL — the generic router's ``POST /api/integrations/{name}/connect``
covers it (see ``integrations/views.py`` and ``integrations/mcp/oauth_views.py``).
"""

from __future__ import annotations

from django.urls import path

from django_ai_sdk.integrations.mcp import oauth_views

app_name = "integrations_mcp"

urlpatterns = [
    path("oauth/<str:server_name>/callback/", oauth_views.oauth_callback, name="oauth-callback"),
]
