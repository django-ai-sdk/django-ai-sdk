from __future__ import annotations

from django.urls import path

from django_ai_sdk.integrations.webhooks import views

app_name = "integrations_webhooks"

# Keep the path stable: platforms store it.
urlpatterns = [
    path("<str:name>/webhook/", views.receive, name="webhook"),
]
