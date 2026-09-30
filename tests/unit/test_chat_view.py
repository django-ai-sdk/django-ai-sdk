"""ChatView through real URLs: with a thread, and stateless with a fixed agent."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from django.http import HttpResponse

from django_ai_sdk.permissions import PermissionDenied


def post(client, body):
    return client.post("/chat/t1/", data=body, content_type="application/json")


VALID = json.dumps({"messages": [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]})


@pytest.mark.django_db
class TestChatView:
    def test_a_malformed_body_is_an_invalid_request(self, client):
        response = post(client, "{not json")
        assert response.status_code == 400
        assert response.json()["code"] == "invalid_request"

    def test_an_unknown_thread_is_not_found(self, client):
        response = post(client, VALID)
        assert response.status_code == 404
        assert response.json()["code"] == "not_found"

    def test_a_refusal_is_forbidden(self, client):
        with patch(
            "django_ai_sdk.agents.services.AgentService.get_agent",
            AsyncMock(side_effect=PermissionDenied("nope")),
        ):
            assert post(client, VALID).status_code == 403

    def test_a_crash_answers_with_a_code_not_its_text(self, client):
        with patch(
            "django_ai_sdk.agents.services.AgentService.get_agent",
            AsyncMock(side_effect=RuntimeError("/var/www/SECRET")),
        ):
            response = post(client, VALID)
        assert response.status_code == 500
        assert "SECRET" not in response.content.decode()

    def test_the_agent_answers_with_its_own_response(self, client):
        agent = AsyncMock()
        agent.as_view.return_value = HttpResponse("streamed")
        with patch(
            "django_ai_sdk.agents.services.AgentService.get_agent", AsyncMock(return_value=agent)
        ):
            response = post(client, VALID)
        assert response.content == b"streamed"
        assert agent.as_view.await_args.kwargs["thread_id"] == "t1"

    def test_only_post_is_allowed(self, client):
        assert client.get("/chat/t1/").status_code == 405


@pytest.mark.django_db
class TestStatelessChat:
    def test_the_views_agent_answers_without_a_thread(self, client, mock_agents_registry):
        agent = mock_agents_registry.get.return_value
        agent.as_view = AsyncMock(return_value=HttpResponse("streamed"))
        body = json.dumps({**json.loads(VALID), "agent_id": "someone-else"})

        response = client.post("/stateless-chat/", data=body, content_type="application/json")

        assert response.content == b"streamed"
        # The URL's agent, never the payload's; and no thread, so nothing is stored.
        mock_agents_registry.get.assert_called_with("test-agent")
        assert agent.as_view.await_args.kwargs["thread_id"] is None

    def test_without_an_agent_there_is_nothing_to_chat_with(self, client):
        response = client.post("/agentless-chat/", data=VALID, content_type="application/json")
        assert response.status_code == 404
        assert response.json()["code"] == "not_found"
