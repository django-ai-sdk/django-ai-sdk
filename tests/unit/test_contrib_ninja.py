"""The contrib Ninja layer as an HTTP caller sees it, through tests/urls.py.

Services have their own tests; these prove the wiring: the request's user reaches the
service, errors come back as the right status with an error code, and the router
factories can be trimmed and mounted more than once.
"""

import json

import pytest
from ninja import NinjaAPI

from django_ai_sdk.contrib import ninja as ai_sdk_routers
from tests.urls import api

# DRF serves the same workflow endpoints, so the workflow tests run on both mounts.
MOUNTS = {"ninja": "/api/workflows/", "drf": "/drf/workflows/"}
VALID = {"name": "acl-wf", "steps": [{"name": "result", "agent_id": "a"}]}
NO_STEPS = {"name": "empty-wf", "steps": []}


def post_json(client, url, body):
    return client.post(url, data=json.dumps(body), content_type="application/json")


@pytest.fixture(params=list(MOUNTS.values()), ids=list(MOUNTS))
def workflows(request):
    return request.param


def created_id(client, workflows, name="acl-wf"):
    response = post_json(client, workflows, {"name": name, "workflow": VALID})
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
        other.add_router("/", ai_sdk_routers.get_threads_router())
        assert "/threads/" in other.get_openapi_schema(path_prefix="")["paths"]

    def test_excluded_endpoints_are_left_out(self):
        other = NinjaAPI(urls_namespace="trimmed")
        other.add_router("/", ai_sdk_routers.get_threads_router(exclude={"delete_all_threads"}))
        methods = other.get_openapi_schema(path_prefix="")["paths"]["/threads/"]
        assert "delete" not in methods and "get" in methods

    def test_a_misspelt_exclude_fails_loudly(self):
        with pytest.raises(ValueError, match="delete_all_thread"):
            ai_sdk_routers.get_threads_router(exclude={"delete_all_thread"})


@pytest.mark.django_db(transaction=True)
class TestThreads:
    def test_an_anonymous_caller_is_turned_away(self, client):
        assert client.get("/api/threads/").status_code == 401

    def test_an_unknown_thread_is_not_found_with_a_code(self, client, users):
        client.force_login(users[0])
        response = client.delete("/api/threads/00000000-0000-0000-0000-000000000000/")
        assert response.status_code == 404
        assert response.json()["code"] == "not_found"

    def test_pages_are_bounded(self, client, users):
        """Services take limit=None (everything); over HTTP a page is at most 100."""
        client.force_login(users[0])
        assert client.get("/api/threads/?limit=100").status_code == 200
        assert client.get("/api/threads/?limit=101").status_code == 422
        assert client.get("/api/threads/?offset=-1").status_code == 422

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
    def test_an_author_creates_and_reads_back_their_own(self, client, users, workflows):
        client.force_login(users[0])
        workflow_id = created_id(client, workflows)

        assert client.get(f"{workflows}{workflow_id}/").status_code == 200
        assert [w["id"] for w in client.get(workflows).json()] == [workflow_id]

    def test_without_the_manage_permission_a_workflow_is_absent(self, client, users, workflows):
        owner, stranger, staff = users
        client.force_login(owner)
        workflow_id = created_id(client, workflows)

        for user in (stranger, staff):
            client.force_login(user)
            # 404 and not 403 on every door, because a 403 would confirm the id exists.
            assert client.get(f"{workflows}{workflow_id}/").status_code == 404
            assert client.delete(f"{workflows}{workflow_id}/").status_code == 404
            patch = client.patch(
                f"{workflows}{workflow_id}/",
                data=json.dumps({"name": "stolen"}),
                content_type="application/json",
            )
            assert patch.status_code == 404
            assert client.get(workflows).status_code == 403
            assert post_json(client, workflows, {"name": "x", "workflow": VALID}).status_code == 403

    def test_a_signed_in_user_runs_a_stored_workflow(self, client, users, workflows):
        from unittest.mock import patch

        owner, stranger, _ = users
        client.force_login(owner)
        workflow_id = created_id(client, workflows)

        client.force_login(stranger)
        with patch("django_ai_sdk.workflows.runs._enqueue"):
            response = post_json(client, f"{workflows}acl-wf/run/", {"inputs": {}})

        assert response.status_code == 202
        runs = client.get(f"{workflows}{workflow_id}/runs/").json()
        assert [(r["id"], r["workflow_key"]) for r in runs] == [
            (response.json()["run_id"], "acl-wf")
        ]
        detail = client.get(f"{workflows}{workflow_id}/runs/{runs[0]['id']}/").json()
        assert detail["workflow_key"] == "acl-wf"

    def test_a_failed_run_resumes_and_a_pending_one_does_not(
        self, client, users, workflows, django_capture_on_commit_callbacks
    ):
        from unittest.mock import patch

        from django_ai_sdk.workflows.models import WorkflowRun

        client.force_login(users[0])
        created_id(client, workflows)
        with (
            patch("django_ai_sdk.workflows.runs._enqueue") as enqueue,
            django_capture_on_commit_callbacks(execute=True),
        ):
            run_id = post_json(client, f"{workflows}acl-wf/run/", {"inputs": {}}).json()["run_id"]
            refused = post_json(client, f"{workflows}runs/{run_id}/resume/", {})
            WorkflowRun.objects.filter(id=run_id).update(status=WorkflowRun.Status.FAILED)
            resumed = post_json(client, f"{workflows}runs/{run_id}/resume/", {})

        assert refused.status_code == 400
        assert resumed.status_code == 202
        assert resumed.json() == {"run_id": run_id, "status": "pending"}
        assert enqueue.call_count == 2

    def test_a_stranger_cannot_resume_a_run(self, client, users, workflows):
        from unittest.mock import patch

        from django_ai_sdk.workflows.models import WorkflowRun

        owner, stranger, _ = users
        client.force_login(owner)
        created_id(client, workflows)
        with patch("django_ai_sdk.workflows.runs._enqueue"):
            run_id = post_json(client, f"{workflows}acl-wf/run/", {"inputs": {}}).json()["run_id"]
        WorkflowRun.objects.filter(id=run_id).update(status=WorkflowRun.Status.FAILED)

        client.force_login(stranger)
        assert post_json(client, f"{workflows}runs/{run_id}/resume/", {}).status_code == 404

    def test_the_listeners_are_listed_only_to_a_workflow_manager(self, client, users, workflows):
        owner, stranger, _ = users
        client.force_login(owner)
        assert client.get(f"{workflows}listeners/").json() == []
        client.force_login(stranger)
        assert client.get(f"{workflows}listeners/").status_code == 403

    def test_an_unknown_key_does_not_run(self, client, users, workflows):
        client.force_login(users[0])

        assert post_json(client, f"{workflows}no-such-workflow/run/", {}).status_code == 404

    def test_a_definition_sent_in_a_request_is_never_run(self, client, users, workflows):
        from django_ai_sdk.workflows.models import WorkflowRun

        client.force_login(users[0])
        before = WorkflowRun.objects.count()

        response = post_json(client, f"{workflows}run/", {"workflow": VALID, "inputs": {}})

        assert response.status_code in (404, 405)
        assert WorkflowRun.objects.count() == before

    def test_a_definition_with_no_steps_is_a_bad_request(self, client, users, workflows):
        client.force_login(users[0])

        response = post_json(client, workflows, {"name": "empty", "workflow": NO_STEPS})

        assert response.status_code == 400
        assert response.json()["code"] == "invalid_request"

    def test_a_run_of_an_unknown_workflow_is_absent(self, client, users, workflows):
        client.force_login(users[0])

        response = post_json(
            client, f"{workflows}00000000-0000-0000-0000-000000000000/run/", {"inputs": {}}
        )

        assert response.status_code == 404
