"""Point Telegram at this deployment's webhook endpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
from django.core.management.base import BaseCommand, CommandError

from django_ai_sdk.integrations.telegram.integration import API_BASE, TelegramIntegration

if TYPE_CHECKING:
    from argparse import ArgumentParser

REQUEST_TIMEOUT_SECONDS = 15


class Command(BaseCommand):
    help = "Register this deployment's telegram webhook endpoint with Telegram."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--url",
            required=True,
            help="Public HTTPS URL of the telegram webhook endpoint.",
        )
        parser.add_argument(
            "--drop-pending",
            action="store_true",
            help="Discard updates that queued up while no webhook was registered.",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        integration = TelegramIntegration()
        bot_token = integration.secret("BOT_TOKEN")
        webhook_secret = integration.secret("WEBHOOK_SECRET")
        missing = [
            key
            for key, value in (("BOT_TOKEN", bot_token), ("WEBHOOK_SECRET", webhook_secret))
            if not value
        ]
        if missing:
            raise CommandError(
                f"Missing {', '.join(missing)} in AI_SDK_INTEGRATIONS['telegram']. The bot "
                f"token comes from @BotFather; the webhook secret is a value you choose."
            )

        response = httpx.post(
            f"{API_BASE}/bot{bot_token}/setWebhook",
            json={
                "url": options["url"],
                "secret_token": webhook_secret,
                # Everything else is an edit, a channel post or a callback the
                # integration ignores, and Telegram queues what it is not sent.
                "allowed_updates": ["message"],
                "drop_pending_updates": options["drop_pending"],
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        # The bot token is in the request URL, so only Telegram's own body is shown.
        body = (
            response.json()
            if response.headers.get("content-type", "").startswith("application/json")
            else {}
        )
        if not body.get("ok"):
            raise CommandError(
                f"Telegram refused the webhook: {body.get('description', response.status_code)}"
            )
        self.stdout.write(self.style.SUCCESS(f"Telegram will deliver to {options['url']}"))
