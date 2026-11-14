"""Slack as a webhook integration: an agent answers mentions and direct messages."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

import httpx
from django.http import JsonResponse

from django_ai_sdk.integrations.slack.verify import is_signed_by_slack
from django_ai_sdk.integrations.webhooks.base import (
    REPLY_TIMEOUT_SECONDS,
    InboundEvent,
    WebhookIntegration,
    json_body,
)

if TYPE_CHECKING:
    from django.http import HttpRequest, HttpResponse

logger = logging.getLogger(__name__)

POST_MESSAGE_URL = "https://slack.com/api/chat.postMessage"

TIMESTAMP_HEADER = "X-Slack-Request-Timestamp"
SIGNATURE_HEADER = "X-Slack-Signature"

# Slack writes a mention as <@U123ABC>. Strip it so the agent reads only the question.
_MENTION = re.compile(r"<@[A-Z0-9]+>")


class SlackIntegration(WebhookIntegration):
    """Answer app mentions and direct messages with the configured agent.

    Needs a Slack app with the app_mentions:read, im:history and chat:write
    scopes, with Event Subscriptions pointed at this integration's path, and:

        AI_SDK_INTEGRATIONS = {"slack": {
            "BOT_TOKEN": env("SLACK_BOT_TOKEN"),
            "SIGNING_SECRET": env("SLACK_SIGNING_SECRET"),
            "AGENT": "myapp.agents.SupportAgent",
        }}
    """

    name = "slack"
    label = "Slack"
    # One thread per Slack thread. A top-level message uses its own ts, so its
    # replies land in the same thread.
    thread_per_ref = True

    def missing_config(self) -> str | None:
        return self._missing_secrets(
            "BOT_TOKEN",
            "SIGNING_SECRET",
            hint="the bot token from the Slack app's Install page, the signing secret from "
            "its Basic Information page.",
        )

    def verify(self, request: HttpRequest) -> bool:
        return is_signed_by_slack(
            self.secret("SIGNING_SECRET"),
            request.headers.get(TIMESTAMP_HEADER, ""),
            request.headers.get(SIGNATURE_HEADER, ""),
            request.body,
        )

    def parse(self, request: HttpRequest) -> InboundEvent | HttpResponse | None:
        body = self._json(request)
        if body is None:
            return None

        # Slack checks the endpoint by asking for its challenge back.
        if body.get("type") == "url_verification":
            return JsonResponse({"challenge": body.get("challenge", "")})

        event = body.get("event") or {}
        if not self._is_answerable(event):
            return None

        # Skip a bare mention so the agent never gets an empty prompt.
        text = _MENTION.sub("", event.get("text", "")).strip()
        if not text:
            return None

        return InboundEvent(
            integration=self.name,
            event_id=body.get("event_id", ""),
            text=text,
            conversation_id=event.get("channel", ""),
            # Reply in the existing thread, or start one under the message.
            thread_ref=event.get("thread_ts") or event.get("ts", ""),
            external_user_id=event.get("user", ""),
            workspace_id=body.get("team_id", ""),
        )

    @staticmethod
    def _is_answerable(event: dict[str, Any]) -> bool:
        # A bot_id means our own reply came back. Answering it would loop forever.
        # A subtype marks an edit, a join or a file share, not a question.
        if event.get("bot_id") or event.get("subtype"):
            return False
        if event.get("type") == "app_mention":
            return True
        return event.get("type") == "message" and event.get("channel_type") == "im"

    async def reply(self, event: InboundEvent, text: str) -> None:
        payload = {
            "channel": event.conversation_id,
            "text": text,
            "thread_ts": event.thread_ref,
        }
        async with httpx.AsyncClient(timeout=REPLY_TIMEOUT_SECONDS) as client:
            response = await client.post(
                POST_MESSAGE_URL,
                json=payload,
                headers={"Authorization": f"Bearer {self.secret('BOT_TOKEN')}"},
            )
        # Slack returns 200 with ok=false on errors like a revoked token,
        # so check the body, not the status code.
        body = json_body(response)
        if not body.get("ok"):
            logger.error(
                "Slack refused a reply to %s: %s", event.conversation_id, body.get("error")
            )
