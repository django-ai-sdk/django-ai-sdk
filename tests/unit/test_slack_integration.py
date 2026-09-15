"""Tests for the shipped Slack integration.

Only the chat.postMessage call is faked; signatures are real HMACs over real bytes,
because a verifier that passes a stubbed signature proves nothing.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest
from django.http import JsonResponse
from django.test import override_settings
from django_ai_sdk.integrations.slack.integration import SlackIntegration
from django_ai_sdk.integrations.slack.verify import MAX_AGE_SECONDS, is_signed_by_slack
from django_ai_sdk.integrations.webhooks.base import InboundEvent

SECRET = "8f742231b10e8888abcd99yyyzzz85a5"
CONFIG = {"slack": {"BOT_TOKEN": "xoxb-test", "SIGNING_SECRET": SECRET, "AGENT": "a.b.C"}}


def _sign(body: bytes, timestamp: str | None = None) -> dict[str, str]:
    timestamp = timestamp or str(int(time.time()))
    digest = hmac.new(
        SECRET.encode(), b"v0:%s:%s" % (timestamp.encode(), body), hashlib.sha256
    ).hexdigest()
    return {
        "HTTP_X_SLACK_REQUEST_TIMESTAMP": timestamp,
        "HTTP_X_SLACK_SIGNATURE": f"v0={digest}",
    }


def _request(rf, payload: dict, *, sign: bool = True, timestamp: str | None = None):
    body = json.dumps(payload).encode()
    headers = _sign(body, timestamp) if sign else {}
    return rf.post(
        "/api/integrations/slack/webhook/",
        data=body,
        content_type="application/json",
        **headers,
    )


def _mention(**overrides):
    event = {
        "type": "app_mention",
        "user": "U024BE7LH",
        "text": "<@U0LAN0Z89> what is the weather",
        "ts": "1700000000.000100",
        "channel": "C024BE91L",
    }
    event.update(overrides)
    return {"type": "event_callback", "event_id": "Ev1", "team_id": "T1", "event": event}


@pytest.fixture
def slack():
    with override_settings(AI_SDK_INTEGRATIONS=CONFIG):
        yield SlackIntegration()


class TestSignatureVerification:
    """A request is answered only if Slack signed it recently."""

    def test_a_real_signature_passes(self, slack, rf):
        assert slack.verify(_request(rf, _mention())) is True

    def test_an_unsigned_request_fails(self, slack, rf):
        assert slack.verify(_request(rf, _mention(), sign=False)) is False

    def test_a_tampered_body_fails(self, slack, rf):
        request = _request(rf, _mention())
        request._body = b'{"type": "event_callback", "event_id": "Evil"}'
        assert slack.verify(request) is False

    def test_a_stale_signature_fails(self, slack, rf):
        old = str(int(time.time()) - MAX_AGE_SECONDS - 1)
        assert slack.verify(_request(rf, _mention(), timestamp=old)) is False

    def test_a_non_numeric_timestamp_fails(self):
        assert is_signed_by_slack(SECRET, "not-a-time", "v0=abc", b"{}") is False

    def test_an_unconfigured_secret_fails(self):
        assert is_signed_by_slack("", "1700000000", "v0=abc", b"{}") is False


class TestParse:
    """parse answers a mention, ignores noise, and replies inline to the handshake."""

    def test_the_handshake_returns_the_challenge(self, slack, rf):
        payload = {"type": "url_verification", "challenge": "3eZbrw1a"}
        response = slack.parse(_request(rf, payload))
        assert isinstance(response, JsonResponse)
        assert json.loads(response.content) == {"challenge": "3eZbrw1a"}

    def test_a_mention_becomes_an_event(self, slack, rf):
        event = slack.parse(_request(rf, _mention()))
        assert isinstance(event, InboundEvent)
        assert event.text == "what is the weather"
        assert event.conversation_id == "C024BE91L"
        assert event.external_user_id == "U024BE7LH"
        assert event.workspace_id == "T1"
        assert event.event_id == "Ev1"

    def test_a_reply_stays_in_its_thread(self, slack, rf):
        payload = _mention(thread_ts="1699999999.000001")
        assert slack.parse(_request(rf, payload)).thread_ref == "1699999999.000001"

    def test_a_top_level_mention_starts_a_thread_under_itself(self, slack, rf):
        assert slack.parse(_request(rf, _mention())).thread_ref == "1700000000.000100"

    def test_the_agents_own_reply_is_ignored(self, slack, rf):
        # Without this the bot answers itself for as long as the workspace exists.
        assert slack.parse(_request(rf, _mention(bot_id="B024BE7LH"))) is None

    def test_an_edit_is_ignored(self, slack, rf):
        assert slack.parse(_request(rf, _mention(subtype="message_changed"))) is None

    def test_a_channel_message_without_a_mention_is_ignored(self, slack, rf):
        payload = _mention(type="message", channel_type="channel")
        assert slack.parse(_request(rf, payload)) is None

    def test_a_direct_message_is_answered(self, slack, rf):
        payload = _mention(type="message", channel_type="im", text="hello there")
        assert slack.parse(_request(rf, payload)).text == "hello there"

    def test_a_body_that_is_not_json_is_ignored(self, slack, rf):
        request = rf.post(
            "/api/integrations/slack/webhook/", data=b"not json", content_type="application/json"
        )
        assert slack.parse(request) is None


class TestReply:
    """The answer is posted back into the conversation it came from."""

    @staticmethod
    def _event():
        return InboundEvent(
            integration="slack",
            event_id="Ev1",
            text="hi",
            conversation_id="C024BE91L",
            thread_ref="1700000000.000100",
        )

    async def test_it_posts_into_the_thread(self, slack):
        post = AsyncMock(return_value=Mock(json=Mock(return_value={"ok": True})))
        with patch("httpx.AsyncClient.post", post):
            await slack.reply(self._event(), "ahoy")
        assert post.await_args.kwargs["json"] == {
            "channel": "C024BE91L",
            "text": "ahoy",
            "thread_ts": "1700000000.000100",
        }
        assert post.await_args.kwargs["headers"]["Authorization"] == "Bearer xoxb-test"

    async def test_a_refused_post_is_logged(self, slack, caplog):
        # Slack answers 200 with ok=false for a revoked token, so the status code
        # alone would report a lost reply as delivered.
        post = AsyncMock(
            return_value=Mock(json=Mock(return_value={"ok": False, "error": "invalid_auth"}))
        )
        with patch("httpx.AsyncClient.post", post):
            await slack.reply(self._event(), "ahoy")
        assert "invalid_auth" in caplog.text


class TestConfiguration:
    """A half-configured Slack app degrades instead of breaking the site."""

    def test_a_fully_configured_integration_has_no_detail(self, slack):
        assert slack.detail is None

    def test_missing_secrets_name_themselves(self):
        with override_settings(AI_SDK_INTEGRATIONS={"slack": {"AGENT": "a.b.C"}}):
            detail = SlackIntegration().detail
        assert "BOT_TOKEN" in detail
        assert "SIGNING_SECRET" in detail

    async def test_a_misconfigured_integration_reports_disconnected(self):
        with override_settings(AI_SDK_INTEGRATIONS={}):
            assert (await SlackIntegration().get_status()).value == "disconnected"

    def test_it_is_a_webhook_kind(self, slack):
        assert slack.kind == "webhook"
