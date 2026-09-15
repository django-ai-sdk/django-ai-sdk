"""Tests for the shipped Telegram integration.

Only the sendMessage call is faked. Telegram signs nothing, so the property under
test is that the shared secret is compared and that an unset one refuses everything.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from django.test import override_settings
from django_ai_sdk.integrations.telegram.integration import (
    MAX_MESSAGE_LENGTH,
    TelegramIntegration,
)
from django_ai_sdk.integrations.webhooks.base import InboundEvent

SECRET = "a-secret-only-telegram-knows"
CONFIG = {
    "telegram": {
        "BOT_TOKEN": "12345:AAtest",
        "WEBHOOK_SECRET": SECRET,
        "ALLOW_FROM": "*",
        "AGENT": "a.b.C",
    }
}


def _request(rf, payload: dict, *, secret: str | None = SECRET):
    headers = {"HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN": secret} if secret is not None else {}
    return rf.post(
        "/api/integrations/telegram/webhook/",
        data=json.dumps(payload).encode(),
        content_type="application/json",
        **headers,
    )


def _update(**overrides):
    message = {
        "message_id": 11,
        "from": {"id": 848151, "is_bot": False, "username": "ada"},
        "chat": {"id": 848151, "type": "private"},
        "date": 1700000000,
        "text": "what is the weather",
    }
    message.update(overrides)
    return {"update_id": 987654321, "message": message}


@pytest.fixture
def telegram():
    with override_settings(AI_SDK_INTEGRATIONS=CONFIG):
        yield TelegramIntegration()


class TestSecretVerification:
    """An update is answered only if it carries the secret given to setWebhook."""

    def test_the_registered_secret_passes(self, telegram, rf):
        assert telegram.verify(_request(rf, _update())) is True

    def test_a_wrong_secret_fails(self, telegram, rf):
        assert telegram.verify(_request(rf, _update(), secret="guess")) is False

    def test_a_missing_header_fails(self, telegram, rf):
        assert telegram.verify(_request(rf, _update(), secret=None)) is False

    def test_an_unconfigured_secret_refuses_a_bare_request(self, rf):
        # Comparing two empty strings would otherwise let every caller in.
        with override_settings(AI_SDK_INTEGRATIONS={"telegram": {"BOT_TOKEN": "x"}}):
            assert TelegramIntegration().verify(_request(rf, _update(), secret=None)) is False


class TestParse:
    """parse answers a message and ignores everything else Telegram sends."""

    def test_a_message_becomes_an_event(self, telegram, rf):
        event = telegram.parse(_request(rf, _update()))
        assert isinstance(event, InboundEvent)
        assert event.text == "what is the weather"
        assert event.event_id == "987654321"
        assert event.conversation_id == "848151"
        assert event.external_user_id == "848151"
        assert event.thread_ref == "11"

    def test_a_group_mention_is_stripped(self, telegram, rf):
        payload = _update(text="@weatherbot what is the weather")
        assert telegram.parse(_request(rf, payload)).text == "what is the weather"

    def test_a_mention_inside_the_question_is_kept(self, telegram, rf):
        # Only a leading address is markup; a name in the middle is what was said.
        payload = _update(text="ask @ada about the weather")
        assert telegram.parse(_request(rf, payload)).text == "ask @ada about the weather"

    def test_the_agents_own_reply_is_ignored(self, telegram, rf):
        payload = _update(**{"from": {"id": 42, "is_bot": True}})
        assert telegram.parse(_request(rf, payload)) is None

    def test_a_message_without_text_is_ignored(self, telegram, rf):
        payload = _update()
        del payload["message"]["text"]
        assert telegram.parse(_request(rf, payload)) is None

    def test_an_edit_is_ignored(self, telegram, rf):
        payload = _update()
        payload["edited_message"] = payload.pop("message")
        assert telegram.parse(_request(rf, payload)) is None

    def test_a_body_that_is_not_json_is_ignored(self, telegram, rf):
        request = rf.post(
            "/api/integrations/telegram/webhook/",
            data=b"not json",
            content_type="application/json",
        )
        assert telegram.parse(request) is None


class TestReply:
    """The answer is sent back into the chat that asked."""

    @staticmethod
    def _event():
        return InboundEvent(
            integration="telegram",
            event_id="987654321",
            text="hi",
            conversation_id="848151",
            thread_ref="11",
        )

    async def test_it_answers_the_message_that_asked(self, telegram):
        post = AsyncMock(return_value=Mock(json=Mock(return_value={"ok": True})))
        with patch("httpx.AsyncClient.post", post):
            await telegram.reply(self._event(), "ahoy")
        assert post.await_args.args[0].endswith("/bot12345:AAtest/sendMessage")
        assert post.await_args.kwargs["json"] == {
            "chat_id": "848151",
            "text": "ahoy",
            "reply_parameters": {"message_id": 11},
        }

    async def test_a_reference_telegram_did_not_issue_is_not_a_reply_target(self, telegram):
        event = self._event()
        event.thread_ref = ""
        post = AsyncMock(return_value=Mock(json=Mock(return_value={"ok": True})))
        with patch("httpx.AsyncClient.post", post):
            await telegram.reply(event, "ahoy")
        assert "reply_parameters" not in post.await_args.kwargs["json"]

    async def test_a_long_answer_is_truncated_to_what_telegram_accepts(self, telegram):
        post = AsyncMock(return_value=Mock(json=Mock(return_value={"ok": True})))
        with patch("httpx.AsyncClient.post", post):
            await telegram.reply(self._event(), "x" * 5000)
        assert len(post.await_args.kwargs["json"]["text"]) == MAX_MESSAGE_LENGTH

    async def test_a_refused_send_is_logged_with_the_reason(self, telegram, caplog):
        # Telegram reports a blocked bot in the body, so the status code loses why.
        refusal = {"ok": False, "error_code": 403, "description": "Forbidden: bot was blocked"}
        post = AsyncMock(return_value=Mock(json=Mock(return_value=refusal)))
        with patch("httpx.AsyncClient.post", post):
            await telegram.reply(self._event(), "ahoy")
        assert "bot was blocked" in caplog.text

    async def test_the_bot_token_is_never_logged(self, telegram, caplog):
        refusal = {"ok": False, "description": "Bad Request: chat not found"}
        post = AsyncMock(return_value=Mock(json=Mock(return_value=refusal)))
        with patch("httpx.AsyncClient.post", post):
            await telegram.reply(self._event(), "ahoy")
        assert "12345:AAtest" not in caplog.text

    async def test_a_body_that_is_not_json_is_reported_by_status(self, telegram, caplog):
        response = Mock(status_code=502, json=Mock(side_effect=ValueError))
        with patch("httpx.AsyncClient.post", AsyncMock(return_value=response)):
            await telegram.reply(self._event(), "ahoy")
        assert "502" in caplog.text


class TestConfiguration:
    """A half-configured bot degrades instead of breaking the site."""

    def test_a_fully_configured_integration_has_no_detail(self, telegram):
        assert telegram.detail is None

    def test_an_undecided_audience_refuses_to_serve(self):
        # Nobody administers entry to a Telegram bot, so leaving this open has to be
        # a choice rather than an oversight.
        config = {"telegram": {**CONFIG["telegram"]}}
        del config["telegram"]["ALLOW_FROM"]
        with override_settings(AI_SDK_INTEGRATIONS=config):
            assert "ALLOW_FROM" in TelegramIntegration().detail

    def test_answering_anyone_is_sayable(self):
        config = {"telegram": {**CONFIG["telegram"], "ALLOW_FROM": "*"}}
        with override_settings(AI_SDK_INTEGRATIONS=config):
            assert TelegramIntegration().detail is None

    def test_a_named_audience_is_sayable(self):
        config = {"telegram": {**CONFIG["telegram"], "ALLOW_FROM": ["123456789"]}}
        with override_settings(AI_SDK_INTEGRATIONS=config):
            assert TelegramIntegration().detail is None

    def test_missing_secrets_name_themselves(self):
        with override_settings(AI_SDK_INTEGRATIONS={"telegram": {"AGENT": "a.b.C"}}):
            detail = TelegramIntegration().detail
        assert "BOT_TOKEN" in detail
        assert "WEBHOOK_SECRET" in detail

    async def test_a_misconfigured_integration_reports_disconnected(self):
        with override_settings(AI_SDK_INTEGRATIONS={}):
            assert (await TelegramIntegration().get_status()).value == "disconnected"

    def test_it_is_a_webhook_kind(self, telegram):
        assert telegram.kind == "webhook"
