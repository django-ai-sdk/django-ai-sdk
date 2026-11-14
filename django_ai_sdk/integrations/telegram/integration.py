"""Telegram as a webhook integration: an agent answers messages sent to a bot."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

import httpx

from django_ai_sdk.integrations.telegram.verify import is_sent_by_telegram
from django_ai_sdk.integrations.webhooks.base import (
    REPLY_TIMEOUT_SECONDS,
    UNSIGNED_DEDUP_TTL,
    InboundEvent,
    WebhookIntegration,
    json_body,
)

if TYPE_CHECKING:
    from django.http import HttpRequest, HttpResponse

logger = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org"

# Telegram rejects a longer message.
MAX_MESSAGE_LENGTH = 4096

SECRET_HEADER = "X-Telegram-Bot-Api-Secret-Token"

# In a group the bot is addressed as @botname. Strip it so the agent reads the question.
_MENTION = re.compile(r"^@\w+\s*")


class TelegramIntegration(WebhookIntegration):
    """Answer messages sent to a Telegram bot with the configured agent.

    Needs a bot from @BotFather, registered once with setWebhook (the url, and
    WEBHOOK_SECRET as secret_token), and:

        AI_SDK_INTEGRATIONS = {"telegram": {
            "BOT_TOKEN": env("TELEGRAM_BOT_TOKEN"),
            "WEBHOOK_SECRET": env("TELEGRAM_WEBHOOK_SECRET"),
            "ALLOW_FROM": ["123456789"],
            "AGENT": "myapp.agents.SupportAgent",
        }}
    """

    name = "telegram"
    label = "Telegram"

    dedup_ttl = UNSIGNED_DEDUP_TTL

    def missing_config(self) -> str | None:
        missing = self._missing_secrets(
            "BOT_TOKEN",
            "WEBHOOK_SECRET",
            hint="the bot token comes from @BotFather, the webhook secret is a value you "
            "choose and pass to setWebhook as secret_token.",
        )
        if missing:
            return missing
        # Anyone who knows the bot's name can message it, so require an explicit
        # audience.
        if not self.has_audience():
            return (
                f"No audience. Set AI_SDK_INTEGRATIONS['telegram']['ALLOW_FROM'] to the "
                f"Telegram user ids that may ask, or to {self.ANY!r} to answer anyone who "
                f"messages the bot."
            )
        return None

    def verify(self, request: HttpRequest) -> bool:
        return is_sent_by_telegram(
            self.secret("WEBHOOK_SECRET"), request.headers.get(SECRET_HEADER, "")
        )

    def parse(self, request: HttpRequest) -> InboundEvent | HttpResponse | None:
        update = self._json(request)
        if update is None:
            return None

        # Answer only new messages. Edits, channel posts and callback queries use
        # other keys.
        message = update.get("message") or {}
        if not self._is_answerable(message):
            return None

        chat = message.get("chat") or {}
        sender = message.get("from") or {}
        return InboundEvent(
            integration=self.name,
            event_id=str(update.get("update_id", "")),
            text=_MENTION.sub("", message.get("text", "")).strip(),
            conversation_id=str(chat.get("id", "")),
            # Reply to the message itself, so the answer is clear in a busy group
            # and lands in the right forum topic.
            thread_ref=str(message.get("message_id", "")),
            external_user_id=str(sender.get("id", "")),
        )

    @staticmethod
    def _is_answerable(message: dict[str, Any]) -> bool:
        # is_bot means our own reply came back. Answering it would loop forever.
        # A message without text is a sticker, a photo or a join, not a question.
        if (message.get("from") or {}).get("is_bot"):
            return False
        return bool(message.get("text"))

    async def send(self, chat_id: str, text: str, *, reply_to: str | None = None) -> bool:
        """Send a message to a chat, with or without an inbound event.

        Returns whether Telegram accepted it. Refusals are logged, not raised.
        """
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text[:MAX_MESSAGE_LENGTH],
        }
        # Telegram message ids are numbers. Anything else is ignored, so the
        # message still goes out, just not as a reply.
        if reply_to and str(reply_to).isdigit():
            payload["reply_parameters"] = {"message_id": int(reply_to)}

        async with httpx.AsyncClient(timeout=REPLY_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{API_BASE}/bot{self.secret('BOT_TOKEN')}/sendMessage", json=payload
            )
        # Read ok and description from the body, which says why a send failed.
        body = json_body(response)
        if not body.get("ok"):
            logger.error(
                "Telegram refused a send to %s: %s",
                chat_id,
                body.get("description", response.status_code),
            )
            return False
        return True

    async def reply(self, event: InboundEvent, text: str) -> None:
        await self.send(event.conversation_id, text, reply_to=event.thread_ref or None)
