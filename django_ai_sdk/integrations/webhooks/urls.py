"""Receive endpoint for webhook integrations.

Include it in the host project's URLconf:

    path("api/integrations/", include("django_ai_sdk.integrations.webhooks.urls")),

Slack's Event Subscriptions URL is then
https://example.com/api/integrations/slack/webhook/. Keep the path stable, since
the platform stores it.
"""

from __future__ import annotations

from django.urls import path

from django_ai_sdk.integrations.webhooks import views

app_name = "integrations_webhooks"

urlpatterns = [
    path("<str:name>/webhook/", views.receive, name="webhook"),
]
