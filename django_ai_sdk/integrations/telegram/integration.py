"""Telegram as a webhook integration: an agent answers messages sent to a bot."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

import httpx
from django.utils.crypto import constant_time_compare

from django_ai_sdk.integrations.webhooks.base import InboundEvent, WebhookIntegration

if TYPE_CHECKING:
    from django.http import HttpRequest, HttpResponse

logger = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org"
SEND_TIMEOUT_SECONDS = 10

# Telegram rejects a message longer than this outright.
MAX_MESSAGE_LENGTH = 4096

SECRET_HEADER = "X-Telegram-Bot-Api-Secret-Token"

# In a group the bot is addressed as @botname; the agent should read what follows.
_LEADING_MENTION = re.compile(r"^@\w+\s*")


class TelegramIntegration(WebhookIntegration):
    """Answers messages sent to a Telegram bot as the configured agent.

    Needs a bot from @BotFather, `manage.py set_telegram_webhook --url ...` run once,
    and:

        AI_SDK_INTEGRATIONS = {"telegram": {
            "BOT_TOKEN": env("TELEGRAM_BOT_TOKEN"),
            "WEBHOOK_SECRET": env("TELEGRAM_WEBHOOK_SECRET"),
            "ALLOW_FROM": ["123456789"],
            "AGENT": "myapp.agents.SupportAgent",
        }}
    """

    name = "telegram"
    label = "Telegram"

    def missing_config(self) -> str | None:
        missing = [key for key in ("BOT_TOKEN", "WEBHOOK_SECRET") if not self.secret(key)]
        if missing:
            return (
                f"Missing {', '.join(missing)}. Set them in AI_SDK_INTEGRATIONS['telegram']: "
                f"the bot token comes from @BotFather, the webhook secret is a value you "
                f"choose and pass to set_telegram_webhook."
            )
        # Slack and Discord are entered through an installation someone administers.
        # A Telegram bot is reachable by anyone who knows its name, so the audience is
        # named here or nothing is answered.
        if not self.has_audience():
            return (
                f"No audience. Set AI_SDK_INTEGRATIONS['telegram']['ALLOW_FROM'] to the "
                f"Telegram user ids that may ask, or to {self.ANY!r} to answer anyone who "
                f"messages the bot."
            )
        return None

    def verify(self, request: HttpRequest) -> bool:
        # Telegram signs nothing; it echoes back the secret registered with setWebhook.
        # An unset secret is refused outright, since it would match a request that
        # sends no header at all.
        secret = self.secret("WEBHOOK_SECRET")
        return bool(secret) and constant_time_compare(
            request.headers.get(SECRET_HEADER, ""), secret
        )

    def parse(self, request: HttpRequest) -> InboundEvent | HttpResponse | None:
        try:
            update = json.loads(request.body)
        except ValueError:
            logger.warning("Telegram sent a body that is not JSON")
            return None

        # Only a new message is answered: an edit, a channel post or a callback query
        # arrives under a different key and would otherwise answer twice.
        message = update.get("message") or {}
        if not self._is_answerable(message):
            return None

        chat = message.get("chat") or {}
        sender = message.get("from") or {}
        return InboundEvent(
            integration=self.name,
            event_id=str(update.get("update_id", "")),
            text=_LEADING_MENTION.sub("", message.get("text", "")).strip(),
            conversation_id=str(chat.get("id", "")),
            # Answering the message that asked keeps the reply readable in a busy
            # group, and puts it in the right topic in a forum.
            thread_ref=str(message.get("message_id", "")),
            external_user_id=str(sender.get("id", "")),
        )

    @staticmethod
    def _is_answerable(message: dict[str, Any]) -> bool:
        # is_bot on the sender is the agent's own reply coming back, which would
        # otherwise answer itself forever. A message with no text is a sticker, a
        # photo or someone joining rather than a question.
        if (message.get("from") or {}).get("is_bot"):
            return False
        return bool(message.get("text"))

    async def reply(self, event: InboundEvent, text: str) -> None:
        payload: dict[str, Any] = {
            "chat_id": event.conversation_id,
            "text": text[:MAX_MESSAGE_LENGTH],
        }
        # Telegram numbers its messages, so a reference it did not produce is not a
        # reply target; the answer lands in the chat instead of failing to send.
        if event.thread_ref.isdigit():
            payload["reply_parameters"] = {"message_id": int(event.thread_ref)}

        async with httpx.AsyncClient(timeout=SEND_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{API_BASE}/bot{self.secret('BOT_TOKEN')}/sendMessage", json=payload
            )
        # Telegram answers with ok=false and a description for a blocked bot or a
        # chat it was removed from, so the status code alone loses the reason.
        body = _as_json(response)
        if not body.get("ok"):
            logger.error(
                "Telegram refused a reply to %s: %s",
                event.conversation_id,
                body.get("description", response.status_code),
            )


def _as_json(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}
