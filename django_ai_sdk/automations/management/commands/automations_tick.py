"""One tick of the scheduler: claim everything due and hand it to the queue.

Exits 0 when the tick completes. A failing run is recorded on its WorkflowRun.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand, CommandError

if TYPE_CHECKING:
    from argparse import ArgumentParser


class Command(BaseCommand):
    help = "Dispatch every automation that is due, or one automation now by name"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "name",
            nargs="?",
            default="",
            help="Dispatch this automation now, outside its schedule",
        )
        parser.add_argument(
            "--loop",
            type=int,
            default=0,
            metavar="SECONDS",
            help="Keep ticking, this many seconds apart",
        )

    def handle(self, *args: object, **options: Any) -> None:
        name, loop = options["name"], options["loop"]
        if name:
            asyncio.run(self._dispatch_now(name))
            return
        while True:
            asyncio.run(self._tick(verbosity=options["verbosity"]))
            if not loop:
                return
            time.sleep(loop)

    async def _dispatch_now(self, name: str) -> None:
        from django_ai_sdk.automations.runner import dispatch_now

        try:
            run = await dispatch_now(name)
        except ValueError as exc:
            raise CommandError(str(exc)) from exc
        if run is not None:
            self.stdout.write(self.style.SUCCESS(f"→ {name}: dispatched run {run.id}"))

    async def _tick(self, *, verbosity: int) -> None:
        from django_ai_sdk.automations.runner import tick

        for result in await tick():
            if result.run is not None:
                self.stdout.write(
                    self.style.SUCCESS(f"→ {result.name}: dispatched run {result.run.id}")
                )
            elif result.reason and verbosity >= 2:
                self.stdout.write(f"  {result.name}: {result.reason}")
