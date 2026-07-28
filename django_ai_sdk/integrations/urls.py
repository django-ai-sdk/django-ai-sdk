"""URLconf for the generic integrations HTTP surface.

Include this in the host project's URLconf, e.g.::

    path("api/integrations/", include("django_ai_sdk.integrations.urls")),

alongside ``django_ai_sdk.integrations.mcp.urls`` for the OAuth callback:

    path("api/integrations/", include("django_ai_sdk.integrations.mcp.urls")),
"""

from __future__ import annotations

from django.urls import path

from django_ai_sdk.integrations import views

urlpatterns = [
    path("", views.list_integrations, name="list"),
    path("<str:name>/connect", views.connect, name="connect"),
    path("<str:name>/disconnect", views.disconnect, name="disconnect"),
    path("<str:name>/reconnect", views.reconnect, name="reconnect"),
]
