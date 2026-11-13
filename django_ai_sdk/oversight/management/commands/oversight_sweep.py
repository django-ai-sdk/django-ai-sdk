"""Clean up turns a crash left behind. Schedule it:

    manage.py oversight_sweep --stale-after 30m --reflect-after 10m --limit 500 [--dry-run]

A turn still open after --stale-after (its worker was killed, or the response never
read) is marked `interrupted`; its outcome stays blank unless a rule already acted.
A turn whose reflect task was lost (never queued, or its worker's lease ran out) is
queued again, oldest first; one that died on its last attempt is marked failed.
Each run handles at most --limit turns of each kind, oldest first.
"""

from __future__ import annotations

import re
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from django_ai_sdk.oversight.persist import enqueue_reflect
from django_ai_sdk.oversight.reflect import REFLECT_ATTEMPTS, claimable_q, mark_reflect_failed

if TYPE_CHECKING:
    from argparse import ArgumentParser

UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}
AGE = re.compile(r"^\s*(\d+)\s*([smhd]?)\s*$")


def parse_age(text: str) -> timedelta:
    """Parse an age like "30m" (s, m, h or d). A bare number means minutes."""
    match = AGE.match(str(text))
    if match is None:
        raise ValueError(f"Expected an age such as 90s, 30m, 2h or 1d, got {text!r}")
    amount, unit = match.groups()
    return timedelta(**{UNITS[unit or "m"]: int(amount)})


class Command(BaseCommand):
    help = (
        "Mark turns still open after --stale-after as interrupted, and queue reflection "
        "again for turns pending longer than --reflect-after or held by a lapsed lease."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--stale-after",
            type=parse_age,
            default="30m",
            help="How long a turn may stay open: 90s, 30m, 2h, 1d (default 30m).",
        )
        parser.add_argument(
            "--reflect-after",
            type=parse_age,
            default="10m",
            help="How long reflection may stay pending before it is queued again (default 10m).",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=500,
            help="The most turns to take in one run (default 500).",
        )
        parser.add_argument(
            "--dry-run", action="store_true", help="Only count the turns; change nothing."
        )

    def handle(self, *args: Any, **options: Any) -> None:
        from django_ai_sdk.oversight.models import Turn

        limit, dry_run = options["limit"], options["dry_run"]
        if limit < 1:
            raise CommandError("--limit must be at least 1")
        now = timezone.now()

        stale = Turn.objects.filter(
            status=Turn.Status.OPEN, created_at__lt=now - options["stale_after"]
        )
        keys = list(stale.order_by("created_at").values_list("pk", flat=True)[:limit])
        # The update repeats the filter, so a turn that closed in the meantime is left alone.
        count = len(keys) if dry_run else stale.filter(pk__in=keys).update(status="interrupted")
        self.stdout.write(f"{_turns(count)} {'would be swept' if dry_run else 'swept'}")

        lost = Turn.objects.filter(claimable_q(now, pending_before=now - options["reflect_after"]))
        keys = list(lost.order_by("created_at").values_list("pk", flat=True)[:limit])
        if dry_run:
            count = len(keys)
        else:
            # These died on their last attempt, so nothing will retry them.
            mark_reflect_failed(
                Turn.objects.filter(
                    reflect_state=Turn.ReflectState.RUNNING,
                    reflect_lease_until__lt=now,
                    reflect_attempts__gte=REFLECT_ATTEMPTS,
                )
            )
            count = sum(enqueue_reflect(key) for key in keys)
        self.stdout.write(f"{_turns(count)} {'would be queued' if dry_run else 'queued'}")


def _turns(count: int) -> str:
    return f"{count} turn{'' if count == 1 else 's'}"
