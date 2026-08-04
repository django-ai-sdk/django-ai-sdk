"""OAuth callback endpoint for MCP integrations.

Include this in the host project's URLconf, e.g.:

    path("api/integrations/", include("django_ai_sdk.integrations.mcp.urls")),

so the reverse name resolves as integrations_mcp:oauth-callback, which is the name
your connect endpoint reverses to build its redirect_uri.

This is the only URL the SDK ships. There is deliberately no oauth-start URL: the
host project's own POST /{name}/connect endpoint, built over
IntegrationService.connect(), covers it (see integrations/services.py and
demo/piratespeak/views_integrations_ninja.py for a complete reference).
"""

from __future__ import annotations

from django.urls import path

from django_ai_sdk.integrations.mcp import oauth_views

app_name = "integrations_mcp"

urlpatterns = [
    path("oauth/<str:server_name>/callback/", oauth_views.oauth_callback, name="oauth-callback"),
]
