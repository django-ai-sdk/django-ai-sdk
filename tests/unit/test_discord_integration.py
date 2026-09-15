"""Tests for the shipped Discord integration.

Only the follow-up edit is faked; signatures are real Ed25519 over real bytes,
because a verifier that passes a stubbed signature proves nothing.
"""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from django.http import JsonResponse
from django.test import override_settings
from django_ai_sdk.integrations.discord.integration import (
    MAX_MESSAGE_LENGTH,
    DiscordIntegration,
)
from django_ai_sdk.integrations.discord.verify import MAX_AGE_SECONDS, is_signed_by_discord
from django_ai_sdk.integrations.webhooks.base import InboundEvent

_PRIVATE_KEY = Ed25519PrivateKey.generate()
PUBLIC_KEY = (
    _PRIVATE_KEY.public_key()
    .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    .hex()
)
CONFIG = {
    "discord": {
        "PUBLIC_KEY": PUBLIC_KEY,
        "APPLICATION_ID": "845027738276462632",
        "AGENT": "a.b.C",
    }
}


def _sign(body: bytes, timestamp: str | None = None) -> dict[str, str]:
    timestamp = timestamp or str(int(time.time()))
    return {
        "HTTP_X_SIGNATURE_TIMESTAMP": timestamp,
        "HTTP_X_SIGNATURE_ED25519": _PRIVATE_KEY.sign(timestamp.encode() + body).hex(),
    }


def _request(rf, payload: dict, *, sign: bool = True, timestamp: str | None = None):
    body = json.dumps(payload).encode()
    headers = _sign(body, timestamp) if sign else {}
    return rf.post(
        "/api/integrations/discord/webhook/",
        data=body,
        content_type="application/json",
        **headers,
    )


def _command(**overrides):
    payload = {
        "id": "1109521900641501205",
        "application_id": "845027738276462632",
        "token": "aW50ZXJhY3Rpb246MTIz",
        "type": 2,
        "channel_id": "1109334398147125288",
        "guild_id": "1109334398147125285",
        "member": {"user": {"id": "846194217717006357"}},
        "data": {
            "name": "ask",
            "options": [{"name": "question", "type": 3, "value": "what is the weather"}],
        },
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def discord():
    with override_settings(AI_SDK_INTEGRATIONS=CONFIG):
        yield DiscordIntegration()


class TestSignatureVerification:
    """An interaction is answered only if Discord signed it recently."""

    def test_a_real_signature_passes(self, discord, rf):
        assert discord.verify(_request(rf, _command())) is True

    def test_an_unsigned_request_fails(self, discord, rf):
        assert discord.verify(_request(rf, _command(), sign=False)) is False

    def test_a_tampered_body_fails(self, discord, rf):
        request = _request(rf, _command())
        request._body = json.dumps(_command(id="evil")).encode()
        assert discord.verify(request) is False

    def test_a_stale_signature_fails(self, discord, rf):
        old = str(int(time.time()) - MAX_AGE_SECONDS - 1)
        assert discord.verify(_request(rf, _command(), timestamp=old)) is False

    def test_a_signature_that_is_not_hex_fails(self):
        assert is_signed_by_discord(PUBLIC_KEY, str(int(time.time())), "zzz", b"{}") is False

    def test_a_public_key_that_is_not_a_key_fails(self):
        assert is_signed_by_discord("abcd", str(int(time.time())), "ab", b"{}") is False

    def test_a_non_numeric_timestamp_fails(self):
        assert is_signed_by_discord(PUBLIC_KEY, "not-a-time", "ab", b"{}") is False

    def test_an_unconfigured_public_key_fails(self):
        assert is_signed_by_discord("", "1700000000", "ab", b"{}") is False


class TestParse:
    """parse answers a command, ignores the rest, and pongs the handshake."""

    def test_the_handshake_returns_a_pong(self, discord, rf):
        response = discord.parse(_request(rf, {"type": 1}))
        assert isinstance(response, JsonResponse)
        assert json.loads(response.content) == {"type": 1}

    def test_a_command_becomes_an_event(self, discord, rf):
        event = discord.parse(_request(rf, _command()))
        assert isinstance(event, InboundEvent)
        assert event.text == "what is the weather"
        assert event.event_id == "1109521900641501205"
        assert event.conversation_id == "1109334398147125288"
        assert event.external_user_id == "846194217717006357"
        assert event.workspace_id == "1109334398147125285"

    def test_a_direct_message_carries_the_speaker_at_the_top_level(self, discord, rf):
        payload = _command(member=None, user={"id": "846194217717006357"})
        assert discord.parse(_request(rf, payload)).external_user_id == "846194217717006357"

    def test_the_interaction_token_survives_on_the_event(self, discord, rf):
        # reply() edits the deferred message with it, so losing it loses the answer.
        event = discord.parse(_request(rf, _command()))
        assert event.reply_token == "aW50ZXJhY3Rpb246MTIz"

    def test_nothing_else_from_the_payload_is_carried(self, discord, rf):
        # Whatever the event holds is written to the queue backend's own storage.
        event = discord.parse(_request(rf, _command()))
        assert set(event.model_dump()) == {
            "integration",
            "event_id",
            "text",
            "conversation_id",
            "thread_ref",
            "external_user_id",
            "workspace_id",
            "reply_token",
        }

    def test_a_button_press_is_ignored(self, discord, rf):
        assert discord.parse(_request(rf, _command(type=3))) is None

    def test_the_named_option_is_the_question(self, discord, rf):
        # A command declaring several strings must not be read by position.
        data = {
            "name": "ask",
            "options": [
                {"name": "language", "type": 3, "value": "nl"},
                {"name": "question", "type": 3, "value": "what is the weather"},
            ],
        }
        event = discord.parse(_request(rf, _command(data=data)))
        assert event.text == "what is the weather"

    def test_a_command_with_no_question_is_ignored(self, discord, rf):
        assert discord.parse(_request(rf, _command(data={"name": "ask"}))) is None

    def test_a_body_that_is_not_json_is_ignored(self, discord, rf):
        request = rf.post(
            "/api/integrations/discord/webhook/",
            data=b"not json",
            content_type="application/json",
        )
        assert discord.parse(request) is None


class TestAcknowledgement:
    """Discord needs a deferred response inside three seconds, not a bare 200."""

    def test_the_ack_defers(self, discord):
        response = discord.ack(Mock())
        assert json.loads(response.content) == {"type": 5}


class TestReply:
    """The answer edits the deferred message the command is waiting on."""

    @staticmethod
    def _event():
        return InboundEvent(
            integration="discord",
            event_id="1109521900641501205",
            text="hi",
            conversation_id="1109334398147125288",
            reply_token="aW50ZXJhY3Rpb246MTIz",
        )

    async def test_it_edits_the_original_response(self, discord):
        request = AsyncMock(return_value=Mock(is_success=True))
        with patch("httpx.AsyncClient.patch", request):
            await discord.reply(self._event(), "ahoy")
        url = request.await_args.args[0]
        assert url.endswith(
            "/webhooks/845027738276462632/aW50ZXJhY3Rpb246MTIz/messages/@original"
        )
        assert request.await_args.kwargs["json"] == {"content": "ahoy"}

    async def test_a_long_answer_is_truncated_to_what_discord_accepts(self, discord):
        request = AsyncMock(return_value=Mock(is_success=True))
        with patch("httpx.AsyncClient.patch", request):
            await discord.reply(self._event(), "x" * 3000)
        assert len(request.await_args.kwargs["json"]["content"]) == MAX_MESSAGE_LENGTH

    async def test_a_refused_edit_is_logged(self, discord, caplog):
        response = Mock(is_success=False, status_code=401, text="401: Unauthorized")
        with patch("httpx.AsyncClient.patch", AsyncMock(return_value=response)):
            await discord.reply(self._event(), "ahoy")
        assert "Unauthorized" in caplog.text

    async def test_the_interaction_token_is_never_logged(self, discord, caplog):
        response = Mock(is_success=False, status_code=404, text="404: Not Found")
        with patch("httpx.AsyncClient.patch", AsyncMock(return_value=response)):
            await discord.reply(self._event(), "ahoy")
        assert "aW50ZXJhY3Rpb246MTIz" not in caplog.text


class TestConfiguration:
    """A half-configured Discord application degrades instead of breaking the site."""

    def test_a_fully_configured_integration_has_no_detail(self, discord):
        assert discord.detail is None

    def test_a_missing_public_key_names_itself(self):
        with override_settings(AI_SDK_INTEGRATIONS={"discord": {"AGENT": "a.b.C"}}):
            assert "PUBLIC_KEY" in DiscordIntegration().detail

    def test_a_missing_application_id_names_itself(self):
        # reply() addresses the follow-up webhook with it, so serving needs it.
        config = {"discord": {**CONFIG["discord"], "APPLICATION_ID": ""}}
        with override_settings(AI_SDK_INTEGRATIONS=config):
            assert "APPLICATION_ID" in DiscordIntegration().detail

    def test_serving_needs_no_bot_token(self, discord):
        # BOT_TOKEN is only read when registering the command.
        assert discord.detail is None

    async def test_a_misconfigured_integration_reports_disconnected(self):
        with override_settings(AI_SDK_INTEGRATIONS={}):
            assert (await DiscordIntegration().get_status()).value == "disconnected"

    def test_it_is_a_webhook_kind(self, discord):
        assert discord.kind == "webhook"
