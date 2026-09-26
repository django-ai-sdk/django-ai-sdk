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

    def test_a_bad_page_is_an_invalid_request(self, client, users):
        client.force_login(users[0])
        response = client.get("/drf/threads/?limit=lots")
        assert response.status_code == 400
        assert response.json()["code"] == "invalid_request"

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
