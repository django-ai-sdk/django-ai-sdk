"""Discord as a webhook integration: an agent answers a slash command."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import httpx
from django.http import JsonResponse

from django_ai_sdk.integrations.discord.verify import is_signed_by_discord
from django_ai_sdk.integrations.webhooks.base import InboundEvent, WebhookIntegration

if TYPE_CHECKING:
    from django.http import HttpRequest, HttpResponse

logger = logging.getLogger(__name__)

API_BASE = "https://discord.com/api/v10"
PATCH_TIMEOUT_SECONDS = 10

# Discord rejects a message longer than this outright.
MAX_MESSAGE_LENGTH = 2000

PING = 1
APPLICATION_COMMAND = 2

# The option register_discord_command declares, and the one the agent answers.
QUESTION_OPTION = "question"

PONG = 1
DEFERRED_MESSAGE = 5


class DiscordIntegration(WebhookIntegration):
    """Answers a slash command as the configured agent.

    Needs a Discord application with its Interactions Endpoint URL pointed at this
    integration's webhook path, a command registered by
    `manage.py register_discord_command`, and:

        AI_SDK_INTEGRATIONS = {"discord": {
            "PUBLIC_KEY": env("DISCORD_PUBLIC_KEY"),
            "APPLICATION_ID": env("DISCORD_APPLICATION_ID"),
            "BOT_TOKEN": env("DISCORD_BOT_TOKEN"),
            "AGENT": "myapp.agents.SupportAgent",
        }}
    """

    name = "discord"
    label = "Discord"

    def missing_config(self) -> str | None:
        # BOT_TOKEN is only read by register_discord_command, so an application that
        # already has its command serves without it.
        missing = [key for key in ("PUBLIC_KEY", "APPLICATION_ID") if not self.secret(key)]
        if not missing:
            return None
        return (
            f"Missing {', '.join(missing)}. Set them in AI_SDK_INTEGRATIONS['discord'] from "
            f"the Discord application's General Information page."
        )

    def verify(self, request: HttpRequest) -> bool:
        return is_signed_by_discord(
            self.secret("PUBLIC_KEY"),
            request.headers.get("X-Signature-Timestamp", ""),
            request.headers.get("X-Signature-Ed25519", ""),
            request.body,
        )

    def parse(self, request: HttpRequest) -> InboundEvent | HttpResponse | None:
        try:
            body = json.loads(request.body)
        except ValueError:
            logger.warning("Discord sent a body that is not JSON")
            return None

        # Discord proves it owns the endpoint by asking for a PONG back in a body.
        if body.get("type") == PING:
            return JsonResponse({"type": PONG})
        if body.get("type") != APPLICATION_COMMAND:
            return None

        text = self._question(body.get("data") or {})
        if not text:
            return None

        # A guild interaction carries the speaker under member, a direct message
        # carries it at the top level.
        user = (body.get("member") or {}).get("user") or body.get("user") or {}
        return InboundEvent(
            integration=self.name,
            event_id=body.get("id", ""),
            text=text,
            conversation_id=body.get("channel_id", ""),
            external_user_id=user.get("id", ""),
            workspace_id=body.get("guild_id", ""),
            reply_token=body.get("token", ""),
        )

    @staticmethod
    def _question(data: dict[str, Any]) -> str:
        options = [o for o in data.get("options") or [] if isinstance(o.get("value"), str)]
        # register_discord_command names its option "question"; a command declaring
        # several strings would otherwise be read by position.
        for option in options:
            if option.get("name") == QUESTION_OPTION:
                return option["value"].strip()
        return options[0]["value"].strip() if options else ""

    def ack(self, event: InboundEvent) -> HttpResponse:
        # Discord marks the command failed unless something answers within three
        # seconds; a deferred response shows the asker a thinking state instead.
        return JsonResponse({"type": DEFERRED_MESSAGE})

    async def reply(self, event: InboundEvent, text: str) -> None:
        # The interaction token authenticates this edit and expires fifteen minutes
        # after the command was used, so no bot token is involved.
        url = (
            f"{API_BASE}/webhooks/{self.secret('APPLICATION_ID')}"
            f"/{event.reply_token}/messages/@original"
        )
        async with httpx.AsyncClient(timeout=PATCH_TIMEOUT_SECONDS) as client:
            response = await client.patch(url, json={"content": text[:MAX_MESSAGE_LENGTH]})
        if not response.is_success:
            logger.error(
                "Discord refused a reply in channel %s: %s %s",
                event.conversation_id,
                response.status_code,
                response.text,
            )
