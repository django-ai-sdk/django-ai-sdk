"""Refresh credentials for every registered integration, and warm its tool cache.

Delegates to each Integration.refresh(): a no-op for integrations without
credentials, and (for MCP OAuth) a proactive refresh of tokens nearing expiry. Run on
a schedule (cron / celery beat).

It then calls get_status(), which primes the cached tool list as a side effect. That
is the difference between a user's first message after a deploy paying a live MCP
connect and paying nothing. It only helps integrations whose cache key isn't
per-user (static/token servers), since an OAuth server's tools are cached per user
and can't be warmed without acting as that user, but those are exactly the ones
every user shares.
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
            integrations = {n: integration for n, integration in integrations.items() if n == name}
            if not integrations:
                self.stdout.write(self.style.WARNING(f"No integration named {name!r}"))
                return 0

        failed = 0
        for n, integration in integrations.items():
            try:
                await integration.refresh()
            except Exception as e:  # noqa: BLE001 — one integration must not stop the rest
                self.stdout.write(self.style.ERROR(f"✗ Error refreshing {n!r}: {e}"))
                logger.exception("Error refreshing integration %r", n)
                failed += 1
                continue

            # Warming is best-effort and reported separately: a dead server here is
            # expected (that's what the circuit breaker is for) and must not make a
            # scheduled run look like a credential failure, or exit non-zero.
            try:
                status = await integration.get_status()
                self.stdout.write(self.style.SUCCESS(f"✓ Refreshed {n!r} (status: {status})"))
            except Exception as e:  # noqa: BLE001
                self.stdout.write(
                    self.style.WARNING(f"✓ Refreshed {n!r}, but could not warm it: {e}")
                )
                logger.warning("Could not warm integration %r", n, exc_info=True)

        return 0 if failed == 0 else 1

    def handle(self, *args: object, integration: str | None = None, **options: object) -> None:
        if asyncio.run(self.handle_async(name=integration)):
            raise CommandError("One or more integration refreshes failed")
