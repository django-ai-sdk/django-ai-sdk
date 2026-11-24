"""Receive endpoint for webhook integrations.

    path("api/integrations/", include("django_ai_sdk.integrations.webhooks.urls")),

Keep the path stable: platforms store it.
"""

from __future__ import annotations

from django.urls import path

from django_ai_sdk.integrations.webhooks import views

app_name = "integrations_webhooks"

urlpatterns = [
    path("<str:name>/webhook/", views.receive, name="webhook"),
]
