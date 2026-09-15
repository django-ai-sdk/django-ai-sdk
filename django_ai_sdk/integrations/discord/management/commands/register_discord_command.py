"""Register the slash command the discord integration answers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
from django.core.management.base import BaseCommand, CommandError

from django_ai_sdk.integrations.discord.integration import (
    API_BASE,
    QUESTION_OPTION,
    DiscordIntegration,
)

if TYPE_CHECKING:
    from argparse import ArgumentParser

STRING_OPTION = 3
REQUEST_TIMEOUT_SECONDS = 15


class Command(BaseCommand):
    help = "Register a global slash command on the configured Discord application."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--name", default="ask", help="Command name, without the slash.")
        parser.add_argument("--description", default="Ask the agent a question.")

    def handle(self, *args: Any, **options: Any) -> None:
        integration = DiscordIntegration()
        application_id = integration.secret("APPLICATION_ID")
        bot_token = integration.secret("BOT_TOKEN")
        missing = [
            key
            for key, value in (("APPLICATION_ID", application_id), ("BOT_TOKEN", bot_token))
            if not value
        ]
        if missing:
            raise CommandError(
                f"Missing {', '.join(missing)} in AI_SDK_INTEGRATIONS['discord']. Both come "
                f"from the Discord application: General Information for the id, Bot for the "
                f"token."
            )

        response = httpx.post(
            f"{API_BASE}/applications/{application_id}/commands",
            json={
                "name": options["name"],
                "description": options["description"],
                "options": [
                    {
                        "name": QUESTION_OPTION,
                        "description": "What to ask.",
                        "type": STRING_OPTION,
                        "required": True,
                    }
                ],
            },
            headers={"Authorization": f"Bot {bot_token}"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if not response.is_success:
            raise CommandError(
                f"Discord refused the command: {response.status_code} {response.text}"
            )
        # A global command can take up to an hour to appear in every guild.
        self.stdout.write(self.style.SUCCESS(f"Registered /{options['name']}"))
