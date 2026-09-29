from __future__ import annotations

import pytest
from ninja import NinjaAPI, Schema
from ninja.testing import TestClient
from rest_framework.exceptions import NotAuthenticated

from django_ai_sdk.contrib.drf import exception_handler
from django_ai_sdk.contrib.ninja import register_error_handlers
from django_ai_sdk.errors import NotFound

SECRET = "/var/www/SECRET"


class Payload(Schema):
    count: int


@pytest.fixture
def client():
    api = NinjaAPI(urls_namespace="error-handlers-test")
    register_error_handlers(api)

    @api.get("/crash")
    def crash(request):
        raise RuntimeError(SECRET)

    @api.get("/missing")
    def missing(request):
        raise NotFound(f"Thread {SECRET} not found")

    @api.post("/validated")
    def validated(request, payload: Payload):
        return {"ok": True}

    return TestClient(api)


class TestNinja:
    def test_a_crash_is_a_500_with_a_code(self, client):
        response = client.get("/crash")

        assert response.status_code == 500
        assert response.json()["code"] == "unknown"
        assert "SECRET" not in response.content.decode()

    def test_not_found(self, client):
        response = client.get("/missing")

        assert response.status_code == 404
        assert response.json()["code"] == "not_found"
        assert "SECRET" not in response.content.decode()

    def test_ninjas_own_validation_still_wins(self, client):
        assert client.post("/validated", json={"count": "x"}).status_code == 422


class TestDrf:
    def test_a_crash_is_a_500_with_a_code(self):
        response = exception_handler(RuntimeError(SECRET), {})

        assert response.status_code == 500
        assert response.data["code"] == "unknown"
        assert "SECRET" not in str(response.data)

    def test_drfs_own_exceptions_keep_drf_handling(self):
        response = exception_handler(NotAuthenticated(), {})

        assert response.status_code == 401
        assert "code" not in response.data
