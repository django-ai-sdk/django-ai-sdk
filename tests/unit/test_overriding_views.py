"""Every recipe in docs/content/overriding-views.md, run for real.

This module is its own URLconf (see the `urls` marker), so each recipe is mounted
exactly as the docs show it.
"""

import json

import pytest
from django.http import JsonResponse
from django.urls import include, path
from ninja import NinjaAPI
from ninja.security import SessionAuth, django_auth_superuser
from ninja.throttling import AuthRateThrottle
from rest_framework import serializers as drf_serializers
from rest_framework.decorators import action
from rest_framework.exceptions import MethodNotAllowed
from rest_framework.response import Response
from rest_framework.routers import DefaultRouter

from django_ai_sdk.contrib import ninja as ai_sdk_routers
from django_ai_sdk.contrib.drf import (
    ApiPagination,
    MemoryViewSet,
    ThreadViewSet,
    WorkflowViewSet,
    exception_handler,
)
from django_ai_sdk.contrib.drf.serializers import MemorySerializer
from django_ai_sdk.contrib.ninja import schemas, threads
from django_ai_sdk.errors import NotFound, error_response
from django_ai_sdk.views.chat import ChatView

pytestmark = pytest.mark.urls(__name__)


# --- Ninja -----------------------------------------------------------------

# Replace one endpoint: exclude it, then wrap the SDK's own function.
threads_router = ai_sdk_routers.get_threads_router(exclude={"list_threads"})


@threads_router.get("/threads/", response=schemas.ThreadListResponse, operation_id="list_threads")
async def list_threads(request, limit: int = 100, offset: int = 0):
    response = await threads.list_threads(request, limit=limit, offset=offset)
    response.threads = [t for t in response.threads if t.message_count > 0]
    return response


# Add an endpoint next to the SDK's.
@threads_router.get("/threads/{thread_id}/export/")
async def export_thread(request, thread_id: str):
    return {"thread_id": thread_id, "format": "markdown"}


api = NinjaAPI(auth=SessionAuth(), urls_namespace="overrides")
ai_sdk_routers.register_error_handlers(api)
api.add_router("/", threads_router)
# Stricter auth for one group of endpoints.
api.add_router("/", ai_sdk_routers.get_agents_router(), auth=django_auth_superuser)
# Throttling per mount.
api.add_router("/", ai_sdk_routers.get_workflows_router(), throttle=AuthRateThrottle("100/h"))


# Your own error shape for one exception, after the SDK's handlers.
@api.exception_handler(NotFound)
def not_found(request, exc):
    return api.create_response(request, {"error": "gone"}, status=404)


# --- DRF -------------------------------------------------------------------


class MyThreadViewSet(ThreadViewSet):
    # Change one action, keep the SDK's logic.
    def list(self, request):
        response = super().list(request)
        response.data = [t for t in response.data if t["message_count"] > 0]
        return response

    # Add an action.
    @action(detail=True, methods=["get"])
    def export(self, request, thread_id=None):
        return Response({"thread_id": thread_id, "format": "markdown"})

    # Remove an extra action: its route disappears.
    delete_all = None

    # Refuse a standard method.
    def destroy(self, request, thread_id=None):
        raise MethodNotAllowed("DELETE")


def my_handler(exc, context):
    response = exception_handler(exc, context)
    if "ref" in response.data:  # an SDK error body, not one of DRF's own
        response["X-Error-Ref"] = response.data["ref"]
    return response


class ErrorShapeThreadViewSet(ThreadViewSet):
    def get_exception_handler(self):
        return my_handler


# Swap a request serializer.
class StrictMemorySerializer(MemorySerializer):
    name = drf_serializers.CharField(min_length=3, max_length=80)


class SmallPages(ApiPagination):
    default_limit = 20
    max_limit = 50


class MyMemoryViewSet(MemoryViewSet):
    serializer_classes = {**MemoryViewSet.serializer_classes, "create": StrictMemorySerializer}
    # Change paging.
    pagination_class = SmallPages


router = DefaultRouter()
router.register("memories", MyMemoryViewSet, basename="memory")
router.register("shaped-threads", ErrorShapeThreadViewSet, basename="shaped-thread")
router.register("threads", MyThreadViewSet, basename="thread")
router.register("workflows", WorkflowViewSet, basename="workflow")


# --- ChatView --------------------------------------------------------------


class MyChatView(ChatView):
    def error_response(self, exc):
        status, body = error_response(exc)
        return JsonResponse({"error": body["code"]}, status=status)


urlpatterns = [
    path("api/", api.urls),
    path("drf/", include(router.urls)),
    path("chat/<str:thread_id>/", MyChatView.as_view()),
]


@pytest.fixture
def user(django_user_model):
    return django_user_model.objects.create(email="owner@example.com")


@pytest.fixture
def empty_thread(user, mock_agents_registry):
    from asgiref.sync import async_to_sync

    from django_ai_sdk.storage.services import ThreadService

    return async_to_sync(ThreadService.create_thread)(agent_id="test-agent", user=user)


@pytest.mark.django_db
class TestNinjaRecipes:
    def test_a_replaced_endpoint_keeps_its_operation_id(self):
        ops = api.get_openapi_schema()["paths"]["/api/threads/"]
        assert ops["get"]["operationId"] == "list_threads"

    def test_a_replaced_endpoint_runs_the_new_code(self, client, user, empty_thread):
        client.force_login(user)
        assert client.get("/api/threads/").json() == {"threads": []}

    def test_an_added_endpoint_is_served(self, client, user):
        client.force_login(user)
        assert client.get("/api/threads/t1/export/").json()["format"] == "markdown"

    def test_a_router_can_get_stricter_auth(self, client, user):
        client.force_login(user)
        assert client.get("/api/agents/").status_code == 401

    def test_a_throttled_mount_still_serves(self, client, user):
        client.force_login(user)
        assert client.get("/api/workflows/").status_code == 200

    def test_your_own_handler_wins_for_its_exception(self, client, user):
        client.force_login(user)
        response = client.delete("/api/threads/00000000-0000-0000-0000-000000000000/")
        assert response.status_code == 404
        assert response.json() == {"error": "gone"}


@pytest.mark.django_db
class TestDrfRecipes:
    def test_an_overridden_action_reuses_super(self, client, user, empty_thread):
        client.force_login(user)
        assert client.get("/drf/threads/").json() == []

    def test_an_added_action_is_routed(self, client, user):
        client.force_login(user)
        assert client.get("/drf/threads/t1/export/").json()["format"] == "markdown"

    def test_a_removed_action_has_no_route(self):
        patterns = [str(url.pattern) for url in router.urls]
        assert any(p.startswith("^shaped-threads/all") for p in patterns)  # still there
        assert not any(p.startswith("^threads/all") for p in patterns)

    def test_a_refused_method_is_not_allowed(self, client, user, empty_thread):
        client.force_login(user)
        assert client.delete(f"/drf/threads/{empty_thread}/").status_code == 405

    def test_errors_can_be_extended(self, client, user):
        client.force_login(user)
        response = client.get("/drf/shaped-threads/nope/")
        assert response.status_code == 404
        assert response["X-Error-Ref"] == response.json()["ref"]

    def test_a_swapped_serializer_validates_the_request(self, client, user):
        client.force_login(user)
        response = client.post("/drf/memories/", {"name": "ab"}, content_type="application/json")
        assert response.status_code == 400
        assert "name" in response.json()

    def test_paging_can_be_narrowed(self, client, user):
        from unittest.mock import patch

        client.force_login(user)
        with patch("django_ai_sdk.memories.services.list_memories", return_value=[]) as listed:
            client.get("/drf/memories/")
            assert listed.call_args.kwargs["limit"] == 20
            client.get("/drf/memories/?limit=500")
            assert listed.call_args.kwargs["limit"] == 50

    def test_other_viewsets_are_registered_unchanged(self, client, user):
        client.force_login(user)
        assert client.get("/drf/workflows/").status_code == 200


@pytest.mark.django_db
def test_chat_view_errors_take_your_shape(client, user):
    client.force_login(user)
    response = client.post(
        "/chat/nope/", data=json.dumps({"messages": []}), content_type="application/json"
    )
    assert response.status_code == 404
    assert response.json() == {"error": "not_found"}
