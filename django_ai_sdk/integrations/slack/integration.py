"""Slack as a webhook integration: an agent answers mentions and direct messages."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

import httpx
from django.http import JsonResponse

from django_ai_sdk.integrations.slack.verify import is_signed_by_slack
from django_ai_sdk.integrations.webhooks.base import InboundEvent, WebhookIntegration

if TYPE_CHECKING:
    from django.http import HttpRequest, HttpResponse

logger = logging.getLogger(__name__)

POST_MESSAGE_URL = "https://slack.com/api/chat.postMessage"
POST_TIMEOUT_SECONDS = 10

# Slack renders a mention as <@U123ABC>; the agent should read the words around it.
_MENTION = re.compile(r"<@[A-Z0-9]+>")


class SlackIntegration(WebhookIntegration):
    """Answers app_mention events and direct messages as the configured agent.

    Needs a Slack app with the app_mentions:read, im:history and chat:write scopes,
    its Event Subscriptions URL pointed at this integration's webhook path, and:

        AI_SDK_INTEGRATIONS = {"slack": {
            "BOT_TOKEN": env("SLACK_BOT_TOKEN"),
            "SIGNING_SECRET": env("SLACK_SIGNING_SECRET"),
            "AGENT": "myapp.agents.SupportAgent",
        }}
    """

    name = "slack"
    label = "Slack"

    def missing_config(self) -> str | None:
        missing = [key for key in ("BOT_TOKEN", "SIGNING_SECRET") if not self.secret(key)]
        if not missing:
            return None
        return (
            f"Missing {', '.join(missing)}. Set them in AI_SDK_INTEGRATIONS['slack'] from the "
            f"Slack app's Install page (bot token) and Basic Information page (signing secret)."
        )

    def verify(self, request: HttpRequest) -> bool:
        return is_signed_by_slack(
            self.secret("SIGNING_SECRET"),
            request.headers.get("X-Slack-Request-Timestamp", ""),
            request.headers.get("X-Slack-Signature", ""),
            request.body,
        )

    def parse(self, request: HttpRequest) -> InboundEvent | HttpResponse | None:
        try:
            body = json.loads(request.body)
        except ValueError:
            logger.warning("Slack sent a body that is not JSON")
            return None

        # Slack proves it owns the endpoint by asking for the challenge back in a body.
        if body.get("type") == "url_verification":
            return JsonResponse({"challenge": body.get("challenge", "")})

        event = body.get("event") or {}
        if not self._is_answerable(event):
            return None

        return InboundEvent(
            integration=self.name,
            event_id=body.get("event_id", ""),
            text=_MENTION.sub("", event.get("text", "")).strip(),
            conversation_id=event.get("channel", ""),
            # Answering on thread_ts keeps a reply in its thread; falling back to ts
            # starts one under the message that asked.
            thread_ref=event.get("thread_ts") or event.get("ts", ""),
            external_user_id=event.get("user", ""),
            workspace_id=body.get("team_id", ""),
        )

    @staticmethod
    def _is_answerable(event: dict[str, Any]) -> bool:
        # A bot_id on the event is the agent's own reply coming back, which would
        # otherwise answer itself forever. A subtype marks an edit, a join or a file
        # share rather than someone talking.
        if event.get("bot_id") or event.get("subtype"):
            return False
        if event.get("type") == "app_mention":
            return True
        return event.get("type") == "message" and event.get("channel_type") == "im"

    def thread_scope(self, event: InboundEvent) -> str | None:
        # One SDK thread per Slack thread (channel + thread_ts). Top-level messages
        # use their own ts as thread_ref so the first reply seeds that thread.
        if not event.conversation_id or not event.thread_ref:
            return None
        return f"ai-sdk:webhook:slack:{event.conversation_id}:{event.thread_ref}"

    async def reply(self, event: InboundEvent, text: str) -> None:
        payload = {
            "channel": event.conversation_id,
            "text": text,
            "thread_ts": event.thread_ref,
        }
        async with httpx.AsyncClient(timeout=POST_TIMEOUT_SECONDS) as client:
            response = await client.post(
                POST_MESSAGE_URL,
                json=payload,
                headers={"Authorization": f"Bearer {self.secret('BOT_TOKEN')}"},
            )
        # Slack answers 200 with ok=false for a revoked token or a missing scope, so
        # the status code alone would report a lost reply as delivered.
        body = response.json()
        if not body.get("ok"):
            logger.error(
                "Slack refused a reply to %s: %s", event.conversation_id, body.get("error")
            )
