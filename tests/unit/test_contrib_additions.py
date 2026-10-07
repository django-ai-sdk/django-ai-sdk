"""What the contrib layers add for projects that used to override them."""

from __future__ import annotations

import json

import pytest
from django_ai_sdk.agents.models import AgentSettings, AgentUser
from django_ai_sdk.memories.models import Memory, MemoryUser


@pytest.fixture
def user(django_user_model):
    return django_user_model.objects.create(
        email="owner@example.com", first_name="Olive", last_name="Owner"
    )


def patch_json(client, url, body):
    return client.patch(url, data=json.dumps(body), content_type="application/json")


def make_runtime_agent(user, **fields):
    config = AgentSettings.objects.create(name="Mine", model="m", slug="mine", **fields)
    AgentUser.objects.create(agent=config, user=user, can_manage=True)
    return config


@pytest.mark.django_db(transaction=True)
class TestRuntimeAgents:
    def test_a_runtime_agent_carries_what_the_caller_may_do(self, client, user):
        config = AgentSettings.objects.create(name="Mine", model="m", slug="mine")
        AgentUser.objects.create(agent=config, user=user, can_manage=True)
        client.force_login(user)

        one = client.get(f"/api/agents/runtimes/{config.id}/").json()
        listed = client.get("/api/agents/runtimes/").json()

        assert one["permissions"]["can_manage"] is True
        assert listed[0]["permissions"]["can_manage"] is True

    def test_the_bases_say_what_prompt_they_start_from(self, client, user):
        client.force_login(user)

        bases = client.get("/api/agents/runtimes/bases/").json()

        assert "base_system_prompt" in bases[0]


@pytest.mark.django_db(transaction=True)
class TestMemories:
    def test_memory_users_say_who_they_are(self, client, django_user_model, user):
        memory = Memory.objects.create(name="Shared")
        manager = django_user_model.objects.create(email="manager@x.com")
        MemoryUser.objects.create(memory=memory, user=manager, can_manage=True)
        client.force_login(manager)

        added = client.post(
            f"/api/memories/{memory.id}/users/",
            data=json.dumps({"user_id": str(user.pk)}),
            content_type="application/json",
        ).json()
        listed = client.get(f"/api/memories/{memory.id}/users/").json()

        assert added["email"] == "owner@example.com"
        row = next(u for u in listed if u["email"] == "owner@example.com")
        assert (row["first_name"], row["last_name"]) == ("Olive", "Owner")

    def test_documents_can_be_cancelled_and_tracked_by_task(self, client, user):
        from django_ai_sdk.memories.models import EntryDocument

        memory = Memory.objects.create(name="Mine")
        MemoryUser.objects.create(memory=memory, user=user, can_manage=True)
        doc = EntryDocument.objects.create(
            memory=memory,
            file_name="a.txt",
            task_id="t-1",
            processing_status=EntryDocument.ProcessingStatus.PENDING,
        )
        client.force_login(user)

        cancelled = client.post(f"/api/memories/{memory.id}/documents/{doc.id}/cancel")
        tracked = client.get("/api/memories/tasks/t-1/status")

        assert cancelled.status_code == 200, cancelled.content
        assert cancelled.json()["status"] == "cancelled"
        assert tracked.json()["id"] == str(doc.id)


@pytest.mark.django_db(transaction=True)
class TestRuntimeAgentWrites:
    def test_deleting_answers_with_success(self, client, user):
        config = make_runtime_agent(user)
        client.force_login(user)

        response = client.delete(f"/api/agents/runtimes/{config.id}/")

        assert response.status_code == 200, response.content
        assert response.json()["success"] is True
        assert not AgentSettings.objects.filter(pk=config.pk).exists()


@pytest.mark.django_db(transaction=True)
class TestFixes:
    def test_a_memory_manager_may_delete_it(self, client, django_user_model, user):
        memory = Memory.objects.create(name="Mine", is_public=False)
        MemoryUser.objects.create(memory=memory, user=user, can_manage=True)
        client.force_login(user)

        perms = client.get(f"/api/memories/{memory.id}").json()["permissions"]

        assert perms["can_delete"] is True

    def test_running_an_agent_needs_chat_permission(self, client, django_user_model):
        private = AgentSettings.objects.create(name="Private", model="m", is_public=False)
        outsider = django_user_model.objects.create(email="outsider@x.com")
        client.force_login(outsider)

        response = client.post(
            f"/api/agents/{private.id}/run/",
            data=json.dumps({"messages": [{"id": "1", "role": "user", "parts": []}]}),
            content_type="application/json",
        )

        assert response.status_code == 403, response.content

    def test_a_null_max_history_clears_it(self, client, user):
        config = make_runtime_agent(user, max_history=5)
        client.force_login(user)

        response = patch_json(client, f"/api/agents/runtimes/{config.id}/", {"max_history": None})

        assert response.status_code == 200, response.content
        config.refresh_from_db()
        assert config.max_history is None

    def test_a_null_name_is_refused(self, client, user):
        config = make_runtime_agent(user)
        client.force_login(user)

        response = patch_json(client, f"/api/agents/runtimes/{config.id}/", {"name": None})

        assert response.status_code == 400, response.content
        config.refresh_from_db()
        assert config.name == "Mine"

    def test_an_upload_over_the_limit_is_refused(self, client, user, settings):
        from django.core.files.uploadedfile import SimpleUploadedFile

        from django_ai_sdk.files.common import get_upload_settings

        memory = Memory.objects.create(name="Mine")
        MemoryUser.objects.create(memory=memory, user=user, can_manage=True)
        settings.AI_SDK_MAX_UPLOAD_SIZE = 4
        get_upload_settings.cache_clear()
        client.force_login(user)
        try:
            response = client.post(
                f"/api/memories/{memory.id}/documents",
                {"file": SimpleUploadedFile("big.txt", b"too big")},
            )
        finally:
            get_upload_settings.cache_clear()

        assert response.status_code == 400, response.content
        assert response.json()["code"] == "invalid_request"

    def test_integration_tools_are_not_listed_twice(self, client, user, monkeypatch):
        from types import SimpleNamespace

        from django_ai_sdk.agents.services import AgentService

        async def get_tools():
            return [
                SimpleNamespace(name="own_tool", description="Mine"),
                SimpleNamespace(name="mcp_tool", description="From MCP"),
            ]

        agent = SimpleNamespace(is_runtime=False, config=None, get_tools=get_tools)
        status = SimpleNamespace(
            server_name="mcp", label="MCP", type="mcp", status="ok", tool_names=["mcp_tool"]
        )

        async def get(agent_id):
            return agent

        async def has_perms(*args, **kwargs):
            return True

        async def get_integration_status(agent, user):
            return [status]

        monkeypatch.setattr(AgentService, "get", get)
        monkeypatch.setattr(AgentService, "has_perms", has_perms)
        monkeypatch.setattr(AgentService, "get_integration_status", get_integration_status)
        client.force_login(user)

        body = client.get("/api/agents/any/tools/").json()

        assert [t["label"] for t in body["tools"]] == ["Own Tool"]
        assert body["integrations"][0]["tool_names"] == ["mcp_tool"]
