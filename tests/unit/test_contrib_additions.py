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

