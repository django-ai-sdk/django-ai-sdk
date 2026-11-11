"""Clean up turns a crash left behind. Schedule it:

    manage.py oversight_sweep --stale-after 30m --reflect-after 10m --limit 500 [--dry-run]

A turn still open after --stale-after (its worker was killed, or the response never
read) is marked `interrupted`; its outcome stays blank unless a rule already acted.
A turn whose reflect task was lost (never queued, or its worker's lease ran out) is
queued again, oldest first; one that died on its last attempt is marked failed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from django.core.management.base import CommandError
from django.utils import timezone

from django_ai_sdk.management.sweep import SweepCommand, parse_age

if TYPE_CHECKING:
    from argparse import ArgumentParser
    from datetime import datetime, timedelta

    from django.db.models import QuerySet


class Command(SweepCommand):
    help = (
        "Mark turns still open after --stale-after as interrupted, and queue reflection "
        "again for turns pending longer than --reflect-after or held by a lapsed lease."
    )
    noun = "turn"
    default_stale_after = "30m"
    marked: ClassVar[dict[str, Any]] = {"status": "interrupted"}

    def add_arguments(self, parser: ArgumentParser) -> None:
        super().add_arguments(parser)
        parser.add_argument(
            "--reflect-after",
            default="10m",
            help="How long reflection may stay pending before it is queued again (default 10m).",
        )

    def stale(self, cutoff: datetime) -> QuerySet[Any]:
        from django_ai_sdk.oversight.models import Turn

        return Turn.objects.filter(status=Turn.Status.OPEN, created_at__lt=cutoff)

    def handle(self, *args: Any, **options: Any) -> None:
        try:
            age = parse_age(options["reflect_after"])
        except ValueError as exc:
            raise CommandError(str(exc)) from exc
        super().handle(*args, **options)
        dry_run = bool(options["dry_run"])
        count = self.requeue(age, limit=int(options["limit"]), dry_run=dry_run)
        verb = "would be queued" if dry_run else "queued"
        self.stdout.write(f"{count} turn{'' if count == 1 else 's'} {verb}")

    def requeue(self, age: timedelta, *, limit: int, dry_run: bool) -> int:
        from django_ai_sdk.oversight.models import Turn
        from django_ai_sdk.oversight.persist import enqueue_reflect
        from django_ai_sdk.oversight.reflect import (
            REFLECT_ATTEMPTS,
            claimable_q,
            mark_reflect_failed,
        )

        now = timezone.now()
        lost = Turn.objects.filter(claimable_q(now, pending_before=now - age))
        keys = list(lost.order_by(self.order_by).values_list("pk", flat=True)[:limit])
        if dry_run:
            return len(keys)
        # These died on their last attempt, so nothing will retry them.
        mark_reflect_failed(
            Turn.objects.filter(
                reflect_state=Turn.ReflectState.RUNNING,
                reflect_lease_until__lt=now,
                reflect_attempts__gte=REFLECT_ATTEMPTS,
            )
        )
        return sum(enqueue_reflect(key) for key in keys)
