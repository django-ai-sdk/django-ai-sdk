"""The contrib DRF viewsets through real URLs (tests/urls.py mounts them at /drf/)."""

import json

import pytest

WORKFLOWS = "/drf/workflows/"
VALID = {"name": "acl-wf", "steps": [{"name": "result", "agent_id": "a"}]}
NO_STEPS = {"name": "empty-wf", "steps": []}


@pytest.fixture
def users(django_user_model):
    """An owner, a stranger, and staff."""
    return (
        django_user_model.objects.create(email="owner@example.com"),
        django_user_model.objects.create(email="stranger@example.com"),
        django_user_model.objects.create(email="staff@example.com", is_staff=True),
    )


def post_json(client, url, body):
    return client.post(url, data=json.dumps(body), content_type="application/json")


def created_id(client, name="acl-wf"):
    response = post_json(client, WORKFLOWS, {"name": name, "workflow": VALID})
    assert response.status_code == 201, response.content
    return response.json()["id"]


@pytest.mark.django_db
class TestThreads:
    def test_an_anonymous_caller_is_turned_away_by_the_project_settings(self, client):
        assert client.get("/drf/threads/").status_code == 403

    def test_an_unknown_thread_is_not_found_with_a_code(self, client, users):
        # tests.settings sets no EXCEPTION_HANDLER: the viewsets bring their own.
        client.force_login(users[0])
        response = client.delete("/drf/threads/00000000-0000-0000-0000-000000000000/")
        assert response.status_code == 404
        assert response.json()["code"] == "not_found"

    def test_a_thread_is_created_listed_and_deleted(self, client, users, mock_agents_registry):
        client.force_login(users[0])
        created = post_json(client, "/drf/threads/", {"messages": [], "agent_id": "test-agent"})
        assert created.status_code == 201, created.content
        thread_id = created.json()["thread_id"]

        assert [t["id"] for t in client.get("/drf/threads/").json()] == [thread_id]
        assert client.delete(f"/drf/threads/{thread_id}/").status_code == 204
        assert client.get("/drf/threads/").json() == []

    def test_paging_follows_limit_offset_pagination(self, client, users):
        """DRF's own rules: a bad limit falls back to the default, a big one is capped."""
        from unittest.mock import patch

        client.force_login(users[0])
        with patch("django_ai_sdk.storage.services.list_threads", return_value=[]) as list_threads:
            assert client.get("/drf/threads/?limit=lots").status_code == 200
            assert list_threads.call_args.kwargs["limit"] == 100
            client.get("/drf/threads/?limit=5000&offset=20")
            assert list_threads.call_args.kwargs == {
                "user": users[0],
                "limit": 100,
                "offset": 20,
            }

    def test_chat_is_mounted_next_to_the_viewsets(self, client, users):
        client.force_login(users[0])
        response = post_json(client, "/drf/threads/nope/chat/", {"messages": []})
        assert response.status_code == 404


@pytest.mark.django_db
class TestAgents:
    def test_an_unknown_agent_is_not_found(self, client, users):
        client.force_login(users[0])
        assert client.get("/drf/agents/nope/").status_code == 404

    def test_the_runtime_agent_registries_are_listed(self, client, users):
        client.force_login(users[0])
        assert client.get("/drf/runtime-agents/bases/").status_code == 200
        assert client.get("/drf/runtime-agents/tools/").status_code == 200


@pytest.mark.django_db
class TestWorkflowAccess:
    def test_an_author_creates_and_reads_back_their_own(self, client, users):
        client.force_login(users[0])
        workflow_id = created_id(client)

        assert client.get(f"{WORKFLOWS}{workflow_id}/").status_code == 200
        assert [w["id"] for w in client.get(WORKFLOWS).json()] == [workflow_id]

    def test_a_stranger_finds_someone_elses_workflow_absent(self, client, users):
        owner, stranger, _ = users
        client.force_login(owner)
        workflow_id = created_id(client)

        client.force_login(stranger)

        # 404 and not 403 on every door, because a 403 would confirm the id exists.
        assert client.get(f"{WORKFLOWS}{workflow_id}/").status_code == 404
        assert client.delete(f"{WORKFLOWS}{workflow_id}/").status_code == 404
        assert client.get(WORKFLOWS).json() == []

    def test_staff_read_every_workflow(self, client, users):
        owner, _, staff = users
        client.force_login(owner)
        workflow_id = created_id(client)

        client.force_login(staff)

        assert client.get(f"{WORKFLOWS}{workflow_id}/").status_code == 200

    def test_a_definition_with_no_steps_is_a_bad_request(self, client, users):
        client.force_login(users[0])

        response = post_json(client, WORKFLOWS, {"name": "empty", "workflow": NO_STEPS})

        assert response.status_code == 400
        assert response.json()["code"] == "invalid_request"

    def test_the_registered_actions_are_listed(self, client, users):
        client.force_login(users[0])
        assert client.get(f"{WORKFLOWS}actions/").json() == []


class TestErrorHandler:
    def test_drf_own_errors_keep_drf_handling(self):
        from rest_framework.exceptions import NotFound

        from django_ai_sdk.contrib.drf import exception_handler

        response = exception_handler(NotFound(), {})
        assert response.status_code == 404
        assert "detail" in response.data

    def test_a_service_error_gets_a_code(self):
        from django_ai_sdk.contrib.drf import exception_handler
        from django_ai_sdk.permissions import PermissionDenied

        response = exception_handler(PermissionDenied("no"), {})
        assert response.status_code == 403
        assert response.data["code"] == "permission_denied"


def _pairs():
    from django_ai_sdk.agents.services import AgentSummary
    from django_ai_sdk.contrib.drf import serializers as s
    from django_ai_sdk.integrations.schemas import AgentIntegrationStatus, IntegrationOut
    from django_ai_sdk.memories import schemas as m
    from django_ai_sdk.permissions import ObjectPermissions
    from django_ai_sdk.storage.schemas import ThreadInfo
    from django_ai_sdk.tracing.schemas import TokenUsage, TraceOut
    from django_ai_sdk.views import schemas as v

    return [
        # Responses: the services' own models.
        (s.ThreadSerializer, ThreadInfo),
        (s.MemoryOutSerializer, m.MemoryOut),
        (s.DocumentSerializer, m.DocumentOut),
        (s.DocumentStatusSerializer, m.DocumentStatusOut),
        (s.UploadResultSerializer, m.DocumentUploadResponse),
        (s.ThreadMemorySerializer, m.ThreadMemoryOut),
        (s.MemoryMemberSerializer, m.MemoryUserOut),
        (s.MemoryGroupMemberSerializer, m.MemoryGroupOut),
        (s.TraceSerializer, TraceOut),
        (s.TokenUsageSerializer, TokenUsage),
        (s.IntegrationSerializer, IntegrationOut),
        (s.AgentIntegrationStatusSerializer, AgentIntegrationStatus),
        (s.ObjectPermissionsSerializer, ObjectPermissions),
        (s.AgentSummarySerializer, AgentSummary),
        # Requests: the payloads the Ninja layer validates with.
        (s.ChatRequestSerializer, v.ChatRequest),
        (s.RateMessageSerializer, v.RateMessagePayload),
        (s.AgentSwitchSerializer, v.PatchThreadPayload),
        (s.MemorySerializer, v.MemoryIn),
        (s.BulkConnectSerializer, v.BulkConnectMemoriesIn),
        (s.ToggleActiveSerializer, v.ToggleMemoryActiveIn),
        (s.AddUserSerializer, v.AddMemoryUserIn),
        (s.AddUserSerializer, v.AddAgentUserIn),
        (s.UpdateUserSerializer, v.UpdateMemoryUserIn),
        (s.AddGroupSerializer, v.AddMemoryGroupIn),
        (s.RuntimeAgentCreateSerializer, v.AgentSettingsCreateIn),
        (s.RuntimeAgentUpdateSerializer, v.AgentSettingsUpdateIn),
        (s.WorkflowCreateSerializer, v.WorkflowCreateRequest),
        (s.WorkflowUpdateSerializer, v.WorkflowUpdateRequest),
        (s.WorkflowRunRequestSerializer, v.WorkflowRunRequest),
        (s.WorkflowRunByIdSerializer, v.WorkflowRunByIdRequest),
    ]


@pytest.mark.parametrize(("serializer", "model"), _pairs(), ids=lambda x: x.__name__)
def test_serializers_mirror_the_sdk_models(serializer, model):
    """The DRF layer spells the fields out again; this is what keeps it from drifting."""
    fields = getattr(model, "model_fields", None) or model.__annotations__
    assert set(serializer().fields) == set(fields)
