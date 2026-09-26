"""The studio still boots and serves both APIs with the SDK's current contrib layers."""

import pytest

from demo.urls import api


def test_the_api_combines_sdk_and_studio_endpoints():
    paths = api.get_openapi_schema()["paths"]
    assert "/api/threads/" in paths  # SDK contrib
    assert "/api/threads/{thread_id}/digest/" in paths  # the studio's own


@pytest.mark.django_db
def test_health_answers_a_signed_in_user(client, django_user_model):
    client.force_login(django_user_model.objects.create(email="demo@example.com"))
    assert client.get("/api/health/").json() == {"status": "ok", "service": "django-ai-sdk"}


@pytest.mark.django_db
def test_the_drf_api_is_mounted(client, django_user_model):
    client.force_login(django_user_model.objects.create(email="demo@example.com"))
    assert client.get("/api/v2/workflows/").status_code == 200
