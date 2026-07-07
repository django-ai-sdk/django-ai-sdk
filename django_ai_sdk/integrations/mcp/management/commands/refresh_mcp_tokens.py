"""Refresh expiring OAuth tokens for MCP servers."""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from typing import TYPE_CHECKING

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from django_ai_sdk.integrations.mcp.loader import MCPIntegration, refresh_oauth_token
from django_ai_sdk.integrations.mcp.models import MCPOAuthToken
from django_ai_sdk.integrations.registry import get_all_integrations

if TYPE_CHECKING:
    from argparse import ArgumentParser

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Refresh expiring OAuth tokens for MCP servers"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--threshold",
            type=int,
            default=10,
            help="Minutes before expiry to refresh (default: 10)",
        )
        parser.add_argument(
            "--server",
            type=str,
            help="Refresh only a specific server by name",
        )

    async def handle_async(self, threshold: int, server: str | None) -> int:
        integrations = get_all_integrations()
        oauth_integrations: dict[str, MCPIntegration] = {
            name: i
            for name, i in integrations.items()
            if isinstance(i, MCPIntegration) and i.kind == "oauth"
        }
        if not oauth_integrations:
            self.stdout.write(self.style.WARNING("No OAuth MCP integrations configured"))
            return 0

        now = timezone.now()
        expiry_window = now + timedelta(minutes=threshold)

        query = MCPOAuthToken.objects.filter(expires_at__lte=expiry_window)
        if server:
            query = query.filter(server_name=server)

        tokens = [t async for t in query.all()]

        if not tokens:
            self.stdout.write(self.style.SUCCESS(f"No tokens expiring within {threshold} minutes"))
            return 0

        refreshed = 0
        failed = 0

        for token_obj in tokens:
            server_name = token_obj.server_name
            integration = oauth_integrations.get(server_name)

            if not integration:
                self.stdout.write(
                    self.style.WARNING(
                        f"Server {server_name!r} not found among OAuth integrations"
                    )
                )
                failed += 1
                continue

            try:
                result = await refresh_oauth_token(token_obj, integration.config)
                if result:
                    self.stdout.write(
                        self.style.SUCCESS(
                            f"✓ Refreshed token for {server_name!r} (user {token_obj.user_id})"
                        )
                    )
                    refreshed += 1
                else:
                    self.stdout.write(
                        self.style.WARNING(
                            f"✗ Failed to refresh {server_name!r} "
                            f"(user {token_obj.user_id}) — no refresh token"
                        )
                    )
                    failed += 1
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(
                        f"✗ Error refreshing {server_name!r} (user {token_obj.user_id}): {e}"
                    )
                )
                logger.exception(
                    "Error refreshing token for %s / user %s",
                    server_name,
                    token_obj.user_id,
                )
                failed += 1

        summary = f"Refreshed {refreshed}, failed {failed}/{len(tokens)}"
        if failed == 0:
            self.stdout.write(self.style.SUCCESS(summary))
        else:
            self.stdout.write(self.style.WARNING(summary))

        return 0 if failed == 0 else 1

    def handle(
        self, *args: object, threshold: int, server: str | None, **options: object
    ) -> str | None:
        if asyncio.run(self.handle_async(threshold=threshold, server=server)):
            raise CommandError("One or more token refreshes failed")
        return None
