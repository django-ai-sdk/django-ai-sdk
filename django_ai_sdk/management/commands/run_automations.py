"""One tick of the scheduler: claim everything due and hand it to the queue.

Exits 0 whenever the tick completed; a failing payload is recorded on its own run.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand, CommandError

if TYPE_CHECKING:
    from argparse import ArgumentParser

    from django_ai_sdk.automations.runner import Dispatched

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Dispatch every automation that is currently due"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--automation",
            type=str,
            default="",
            metavar="NAME",
            help="Only consider this automation",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help=(
                "Dispatch regardless of whether it is due. Still respects the lease, so "
                "this cannot start a second copy of a run already in flight."
            ),
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Report what would be dispatched without writing anything",
        )
        parser.add_argument(
            "--loop",
            nargs="?",
            type=int,
            const=60,
            default=None,
            metavar="SECONDS",
            help=(
                "Development only: keep ticking every SECONDS (default 60) instead of "
                "exiting. Not a substitute for a supervised scheduler in production; "
                "nothing restarts this if it dies."
            ),
        )

    def handle(self, *args: object, **options: Any) -> None:
        if options["loop"] and options["dry_run"]:
            raise CommandError("--loop and --dry-run do not make sense together.")

        interval = options["loop"]

        if interval is None:
            asyncio.run(self._tick(options))
            return

        self.stdout.write(
            self.style.WARNING(
                f"Ticking every {interval}s. This is a development convenience; in "
                "production, run this command from cron or your platform's scheduler."
            )
        )
        try:
            while True:
                asyncio.run(self._tick(options))
                time.sleep(interval)
        except KeyboardInterrupt:
            self.stdout.write("\nStopped.")

    async def _tick(self, options: dict[str, Any]) -> None:
        from django_ai_sdk.automations.runner import tick

        results = await tick(
            only=options["automation"],
            force=options["force"],
            dry_run=options["dry_run"],
        )
        self._report(results, verbosity=int(options.get("verbosity", 1)))

    def _report(self, results: list[Dispatched], *, verbosity: int) -> None:
        """Print dispatches; stay silent when nothing was due."""
        for result in results:
            dispatched = [r for r in result.runs if r.status != "skipped"]
            if dispatched:
                self.stdout.write(
                    self.style.SUCCESS(f"→ {result.name}: dispatched {len(dispatched)} run(s)")
                )
            elif result.reason and verbosity >= 2:
                self.stdout.write(f"  {result.name}: {result.reason}")

        if verbosity >= 2 and not results:
            self.stdout.write("No automations are declared.")
