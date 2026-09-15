"""The webhook URL the demo mounts, exercised through the real URLconf.

Everything else calls the view directly, so a broken include or a renamed path would
otherwise only show up against a live Slack app.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time

import pytest
from django.core.cache import cache
from django.test import override_settings
from django.urls import resolve, reverse
from django_ai_sdk.integrations.registry import register, reset_registry
from django_ai_sdk.integrations.slack.integration import SlackIntegration

SECRET = "8f742231b10e8888abcd99yyyzzz85a5"
CONFIG = {
    "slack": {
        "BOT_TOKEN": "xoxb-test",
        "SIGNING_SECRET": SECRET,
        "AGENT": "apps.agents.pirate_basic.PirateBasicAgent",
    }
}


@pytest.fixture(autouse=True)
def _registered():
    """Registered explicitly: other modules reset the process registry on teardown, so
    the app's own ready() cannot be relied on to have survived."""
    reset_registry()
    cache.clear()
    register(SlackIntegration())
    yield
    reset_registry()
    cache.clear()


def test_the_webhook_path_is_named_and_routable():
    url = reverse("integrations_webhooks:webhook", args=["slack"])
    assert url == "/api/integrations/slack/webhook/"
    assert resolve(url).func.__name__ == "receive"


@override_settings(AI_SDK_INTEGRATIONS=CONFIG, ALLOWED_HOSTS=["*"])
async def test_slacks_url_verification_handshake_succeeds(async_client):
    # This is the request Slack sends when the URL is saved; failing it means the
    # event subscription can never be enabled.
    body = json.dumps({"type": "url_verification", "challenge": "3eZbrw1a"}).encode()
    timestamp = str(int(time.time()))
    digest = hmac.new(
        SECRET.encode(), b"v0:%s:%s" % (timestamp.encode(), body), hashlib.sha256
    ).hexdigest()

    response = await async_client.post(
        reverse("integrations_webhooks:webhook", args=["slack"]),
        data=body,
        content_type="application/json",
        headers={"x-slack-request-timestamp": timestamp, "x-slack-signature": f"v0={digest}"},
    )

    assert response.status_code == 200
    assert json.loads(response.content) == {"challenge": "3eZbrw1a"}
