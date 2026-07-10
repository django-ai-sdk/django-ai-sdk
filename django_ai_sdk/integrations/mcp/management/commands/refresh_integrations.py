"""Refresh credentials for every registered integration.

Delegates to each ``IntegrationService.refresh()`` — a no-op for integrations without
credentials, and (for MCP OAuth) a proactive refresh of tokens nearing expiry. Run on a
schedule (cron / celery beat).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from django.core.management.base import BaseCommand, CommandError

from django_ai_sdk.integrations.registry import get_all_integrations

if TYPE_CHECKING:
    from argparse import ArgumentParser

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Refresh credentials for registered integrations (e.g. expiring OAuth tokens)"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--integration",
            type=str,
            help="Refresh only a specific integration by name",
        )

    async def handle_async(self, name: str | None) -> int:
        integrations = await get_all_integrations()
        if name:
            integrations = {n: svc for n, svc in integrations.items() if n == name}
            if not integrations:
                self.stdout.write(self.style.WARNING(f"No integration named {name!r}"))
                return 0

        failed = 0
        for n, svc in integrations.items():
            try:
                await svc.refresh()
                self.stdout.write(self.style.SUCCESS(f"✓ Refreshed {n!r}"))
            except Exception as e:  # noqa: BLE001 — one integration must not stop the rest
                self.stdout.write(self.style.ERROR(f"✗ Error refreshing {n!r}: {e}"))
                logger.exception("Error refreshing integration %r", n)
                failed += 1

        return 0 if failed == 0 else 1

    def handle(self, *args: object, integration: str | None = None, **options: object) -> None:
        if asyncio.run(self.handle_async(name=integration)):
            raise CommandError("One or more integration refreshes failed")
