"""Receive endpoint for webhook integrations.

Include this in the host project's URLconf, beside the MCP OAuth callback:

    path("api/integrations/", include("django_ai_sdk.integrations.webhooks.urls")),

Slack's Event Subscriptions URL is then
https://example.com/api/integrations/slack/webhook/. The path is fixed because a
platform stores it, so it cannot be reverse-built per request the way the host's own
endpoints are.
"""

from __future__ import annotations

from django.urls import path

from django_ai_sdk.integrations.webhooks import views

app_name = "integrations_webhooks"

urlpatterns = [
    path("<str:name>/webhook/", views.receive, name="webhook"),
]
