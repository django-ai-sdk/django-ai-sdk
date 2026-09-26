"""The contrib Ninja layer as an HTTP caller sees it, through tests/urls.py.

Services have their own tests; these prove the wiring: the request's user reaches the
service, errors come back as the right status with an error code, and the router
factories can be trimmed and mounted more than once.
"""

import json

import pytest
from ninja import NinjaAPI

from django_ai_sdk.contrib import ninja as ai
from tests.urls import api

WORKFLOWS = "/api/workflows/"
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


class TestRouterFactories:
    def test_every_router_builds_a_valid_openapi_schema(self):
        assert "/api/threads/" in api.get_openapi_schema()["paths"]

    def test_operation_ids_are_the_function_names(self):
        ops = api.get_openapi_schema()["paths"]["/api/threads/"]
        assert {ops["get"]["operationId"], ops["post"]["operationId"]} == {
            "list_threads",
            "create_thread",
        }

    @pytest.mark.parametrize(
        "path", ["/api/memories/{memory_id}/documents", "/api/memories/thread/{thread_id}/files"]
    )
    def test_uploads_take_a_multipart_file(self, path):
        body = api.get_openapi_schema()["paths"][path]["post"]["requestBody"]
        assert "file" in body["content"]["multipart/form-data"]["schema"]["properties"]

    def test_a_router_can_be_mounted_on_a_second_api(self):
        other = NinjaAPI(urls_namespace="second")
        other.add_router("/", ai.get_threads_router())
        assert "/threads/" in other.get_openapi_schema(path_prefix="")["paths"]

    def test_excluded_endpoints_are_left_out(self):
        other = NinjaAPI(urls_namespace="trimmed")
        other.add_router("/", ai.get_threads_router(exclude={"delete_all_threads"}))
        methods = other.get_openapi_schema(path_prefix="")["paths"]["/threads/"]
        assert "delete" not in methods and "get" in methods

    def test_a_misspelt_exclude_fails_loudly(self):
        with pytest.raises(ValueError, match="delete_all_thread"):
            ai.get_threads_router(exclude={"delete_all_thread"})


@pytest.mark.django_db(transaction=True)
class TestThreads:
    def test_an_anonymous_caller_is_turned_away(self, client):
        assert client.get("/api/threads/").status_code == 401

    def test_an_unknown_thread_is_not_found_with_a_code(self, client, users):
        client.force_login(users[0])
        response = client.delete("/api/threads/00000000-0000-0000-0000-000000000000/")
        assert response.status_code == 404
        assert response.json()["code"] == "not_found"

    def test_a_thread_is_created_listed_and_deleted(self, client, users, mock_agents_registry):
        client.force_login(users[0])
        created = post_json(client, "/api/threads/", {"messages": [], "agent_id": "test-agent"})
        assert created.status_code == 200, created.content
        thread_id = created.json()["thread_id"]

        assert [t["id"] for t in client.get("/api/threads/").json()["threads"]] == [thread_id]

        assert client.delete(f"/api/threads/{thread_id}/").status_code == 200
        assert client.get("/api/threads/").json()["threads"] == []


@pytest.mark.django_db(transaction=True)
class TestAgents:
    def test_reindexing_is_refused_to_anonymous_callers(self, client):
        assert client.post("/api/agents/anything/reindex/").status_code == 401

    def test_an_unknown_agent_is_not_found(self, client, users):
        client.force_login(users[0])
        assert client.get("/api/agents/nope/").status_code == 404


@pytest.mark.django_db(transaction=True)
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
        patch = client.patch(
            f"{WORKFLOWS}{workflow_id}/",
            data=json.dumps({"name": "stolen"}),
            content_type="application/json",
        )
        assert patch.status_code == 404

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

    def test_a_run_of_an_unknown_workflow_is_absent(self, client, users):
        client.force_login(users[0])

        response = post_json(
            client, f"{WORKFLOWS}00000000-0000-0000-0000-000000000000/run/", {"inputs": {}}
        )

        assert response.status_code == 404
