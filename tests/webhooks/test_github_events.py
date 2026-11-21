"""Tests for the shipped GitHub webhook integration.

Only the comment-posting call is faked; signatures are real HMACs over real
bytes, because a verifier that passes a stubbed signature proves nothing.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from django.test import override_settings
from django_ai_sdk.integrations.github_events.integration import (
    MAX_COMMENT_LENGTH,
    GitHubEventsIntegration,
)
from django_ai_sdk.integrations.github_events.verify import is_signed_by_github
from django_ai_sdk.integrations.webhooks.base import InboundEvent

SECRET = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6"
CONFIG = {
    "github_events": {
        "WEBHOOK_SECRET": SECRET,
        "TOKEN": "ghp_test",
        "BOT_USERNAME": "pirate-bot",
        "ALLOW_WORKSPACES": ["acme/widgets"],
        "AGENT": "a.b.C",
    }
}


def _sign(body: bytes) -> dict[str, str]:
    digest = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
    return {"HTTP_X_HUB_SIGNATURE_256": f"sha256={digest}"}


def _request(
    rf,
    payload: dict,
    *,
    event: str = "pull_request",
    sign: bool = True,
    delivery: str = "72d3162e-cc78-11e3-81ab-4c9367dc0958",
):
    body = json.dumps(payload).encode()
    headers = _sign(body) if sign else {}
    headers["HTTP_X_GITHUB_EVENT"] = event
    headers["HTTP_X_GITHUB_DELIVERY"] = delivery
    return rf.post(
        "/api/integrations/github_events/webhook/",
        data=body,
        content_type="application/json",
        **headers,
    )


def _pull_request(**overrides):
    payload = {
        "action": "opened",
        "pull_request": {
            "number": 43,
            "title": "Fix the cannon",
            "body": "Rotates the barrel.",
            "draft": False,
            "html_url": "https://github.com/acme/widgets/pull/43",
            "head": {"sha": "abc123"},
        },
        "repository": {"full_name": "acme/widgets"},
        "sender": {"login": "octocat"},
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def github():
    with override_settings(AI_SDK_INTEGRATIONS=CONFIG):
        yield GitHubEventsIntegration()


class TestSignatureVerification:
    """A request is answered only if GitHub signed it."""

    def test_a_real_signature_passes(self, github, rf):
        assert github.verify(_request(rf, _pull_request())) is True

    def test_an_unsigned_request_fails(self, github, rf):
        assert github.verify(_request(rf, _pull_request(), sign=False)) is False

    def test_a_tampered_body_fails(self, github, rf):
        request = _request(rf, _pull_request())
        request._body = b'{"action": "opened", "pull_request": {"number": 1}}'
        assert github.verify(request) is False

    def test_the_deprecated_sha1_scheme_fails(self, github, rf):
        # sha1 signatures are deprecated; only sha256 is accepted.
        body = json.dumps(_pull_request()).encode()
        digest = hmac.new(SECRET.encode(), body, hashlib.sha1).hexdigest()
        request = rf.post(
            "/api/integrations/github_events/webhook/",
            data=body,
            content_type="application/json",
            HTTP_X_HUB_SIGNATURE=f"sha1={digest}",
            HTTP_X_GITHUB_EVENT="pull_request",
        )
        assert github.verify(request) is False

    def test_a_signature_made_with_another_secret_fails(self, github, rf):
        body = json.dumps(_pull_request()).encode()
        digest = hmac.new(b"another-secret", body, hashlib.sha256).hexdigest()
        request = rf.post(
            "/api/integrations/github_events/webhook/",
            data=body,
            content_type="application/json",
            HTTP_X_HUB_SIGNATURE_256=f"sha256={digest}",
            HTTP_X_GITHUB_EVENT="pull_request",
        )
        assert github.verify(request) is False

    def test_an_unconfigured_secret_fails(self):
        assert is_signed_by_github("", "sha256=abc", b"{}") is False


class TestParse:
    """parse turns a reviewable pull request event into a question, and ignores the rest."""

    def test_the_handshake_is_acknowledged(self, github, rf):
        assert github.parse(_request(rf, {"zen": "Design for failure."}, event="ping")) is None

    def test_an_opened_pull_request_becomes_an_event(self, github, rf):
        event = github.parse(_request(rf, _pull_request()))
        assert isinstance(event, InboundEvent)
        assert event.text.startswith("Pull request #43 in acme/widgets (opened): Fix the cannon")
        assert "Rotates the barrel." in event.text
        assert "Head: abc123" in event.text
        assert event.conversation_id == "acme/widgets"
        assert event.thread_ref == "43"
        assert event.workspace_id == "acme/widgets"
        assert event.external_user_id == "octocat"
        assert event.event_id == hashlib.sha256(json.dumps(_pull_request()).encode()).hexdigest()

    def test_a_new_push_is_reviewed_again(self, github, rf):
        assert github.parse(_request(rf, _pull_request(action="synchronize"))) is not None

    def test_a_closed_pull_request_is_ignored(self, github, rf):
        assert github.parse(_request(rf, _pull_request(action="closed"))) is None

    def test_a_draft_waits_until_it_is_ready(self, github, rf):
        draft = _pull_request()
        draft["pull_request"]["draft"] = True
        assert github.parse(_request(rf, draft)) is None
        draft["action"] = "ready_for_review"
        assert github.parse(_request(rf, draft)) is not None

    def test_the_actions_are_configurable(self, rf):
        config = {"github_events": {**CONFIG["github_events"], "ACTIONS": ["closed"]}}
        with override_settings(AI_SDK_INTEGRATIONS=config):
            github = GitHubEventsIntegration()
            assert github.parse(_request(rf, _pull_request(action="closed"))) is not None
            assert github.parse(_request(rf, _pull_request())) is None

    def test_the_bots_own_push_is_ignored(self, github, rf):
        # Logins are case-insensitive on GitHub.
        payload = _pull_request(sender={"login": "Pirate-Bot"})
        assert github.parse(_request(rf, payload)) is None

    def test_other_events_are_ignored(self, github, rf):
        assert github.parse(_request(rf, {"ref": "refs/heads/main"}, event="push")) is None

    def test_a_body_that_is_not_json_is_ignored(self, github, rf):
        request = rf.post(
            "/api/integrations/github_events/webhook/",
            data=b"not json",
            content_type="application/json",
            HTTP_X_GITHUB_EVENT="pull_request",
        )
        assert github.parse(request) is None

    def test_a_replay_under_a_new_delivery_id_is_a_repeat(self, github, rf):
        # The delivery header is not signed, so the signed body is the dedup key.
        first = github.parse(_request(rf, _pull_request(), delivery="d1"))
        replay = github.parse(_request(rf, _pull_request(), delivery="d2"))
        push = github.parse(_request(rf, _pull_request(action="synchronize"), delivery="d1"))
        assert first.event_id == replay.event_id != push.event_id

    def test_each_pull_request_is_its_own_thread(self, github, rf):
        event = github.parse(_request(rf, _pull_request()))
        assert github.thread_scope(event) == "ai-sdk:webhook:github_events:acme/widgets:43"


class TestReply:
    """The answer is posted as a comment on the pull request."""

    @staticmethod
    def _event():
        return InboundEvent(
            integration="github_events",
            event_id="d1",
            text="hi",
            conversation_id="acme/widgets",
            thread_ref="43",
        )

    async def test_it_posts_a_comment(self, github):
        post = AsyncMock(return_value=Mock(is_success=True))
        with patch("httpx.AsyncClient.post", post):
            await github.reply(self._event(), "ahoy")
        assert post.await_args.args[0] == (
            "https://api.github.com/repos/acme/widgets/issues/43/comments"
        )
        assert post.await_args.kwargs["json"] == {"body": "ahoy"}
        assert post.await_args.kwargs["headers"]["Authorization"] == "Bearer ghp_test"

    async def test_a_long_answer_is_truncated_to_what_github_accepts(self, github):
        post = AsyncMock(return_value=Mock(is_success=True))
        with patch("httpx.AsyncClient.post", post):
            await github.reply(self._event(), "x" * 100_000)
        assert len(post.await_args.kwargs["json"]["body"]) == MAX_COMMENT_LENGTH

    async def test_a_refused_post_is_logged(self, github, caplog):
        response = Mock(is_success=False, status_code=403, text="403: Forbidden")
        with patch("httpx.AsyncClient.post", AsyncMock(return_value=response)):
            await github.reply(self._event(), "ahoy")
        assert "403" in caplog.text

    async def test_the_token_is_never_logged(self, github, caplog):
        response = Mock(is_success=False, status_code=401, text="401: Bad credentials")
        with patch("httpx.AsyncClient.post", AsyncMock(return_value=response)):
            await github.reply(self._event(), "ahoy")
        assert "ghp_test" not in caplog.text


class TestConfiguration:
    """A half-configured webhook degrades instead of breaking the site."""

    def test_a_fully_configured_integration_has_no_detail(self, github):
        assert github.detail is None

    def test_missing_secrets_name_themselves(self):
        with override_settings(AI_SDK_INTEGRATIONS={"github_events": {"AGENT": "a.b.C"}}):
            detail = GitHubEventsIntegration().detail
        assert "WEBHOOK_SECRET" in detail
        assert "TOKEN" in detail

    def test_the_repositories_must_be_named(self):
        # One webhook can cover a whole organization.
        config = {"github_events": {k: v for k, v in CONFIG["github_events"].items() if k != "ALLOW_WORKSPACES"}}
        with override_settings(AI_SDK_INTEGRATIONS=config):
            assert "ALLOW_WORKSPACES" in GitHubEventsIntegration().detail

    async def test_a_misconfigured_integration_reports_disconnected(self):
        with override_settings(AI_SDK_INTEGRATIONS={}):
            assert (await GitHubEventsIntegration().get_status()).value == "disconnected"

    def test_it_is_a_webhook_kind(self, github):
        assert github.kind == "webhook"

    def test_the_replay_bound_is_a_week(self, github):
        # GitHub signs no timestamp, so a captured request replays past verify()
        # once the dedup window has passed: the window is the replay bound.
        assert github.dedup_ttl == 7 * 24 * 60 * 60
