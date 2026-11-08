"""Base for commands that clean up rows a crash left behind.

A row is stale when it is stuck (a turn still open, a run still running) and
older than `--stale-after`. Subclass `SweepCommand`, implement `stale()`, and
set `marked` or override `act()`. Every sweep gets the same options:

    manage.py oversight_sweep --stale-after 30m --limit 500 [--dry-run]

Each run handles at most `--limit` rows, oldest first. Schedule it to work
through a backlog. The default `act()` re-checks staleness in the update, so a
row that moved on in the meantime is left alone.
"""

from __future__ import annotations

import re
from datetime import timedelta
from typing import TYPE_CHECKING, Any, ClassVar

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

if TYPE_CHECKING:
    from argparse import ArgumentParser
    from datetime import datetime

    from django.db.models import QuerySet

UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}
_AGE = re.compile(r"^\s*(\d+)\s*([smhd]?)\s*$")


def parse_age(text: str) -> timedelta:
    """Parse an age like "30m" (s, m, h or d). A bare number means minutes."""
    match = _AGE.match(str(text))
    if match is None:
        raise ValueError(f"Expected an age such as 90s, 30m, 2h or 1d, got {text!r}")
    amount, unit = match.groups()
    return timedelta(**{UNITS[unit or "m"]: int(amount)})


def sweep(
    stale: QuerySet[Any],
    *,
    limit: int,
    order_by: str,
    marked: dict[str, Any],
    dry_run: bool = False,
) -> int:
    """Mark up to `limit` stale rows, oldest first. Return how many were marked.

    The update reuses the `stale` filter, so rows that are no longer stale are
    skipped. With `dry_run`, only count them.
    """
    keys = list(stale.order_by(order_by).values_list("pk", flat=True)[:limit])
    if dry_run or not keys:
        return len(keys)
    return stale.filter(pk__in=keys).update(**marked)


class SweepCommand(BaseCommand):
    """Management command that sweeps stale rows."""

    # Singular name of a row in the output.
    noun: ClassVar[str] = "row"
    default_stale_after: ClassVar[str] = "30m"
    default_limit: ClassVar[int] = 500
    # Field that sorts rows oldest first.
    order_by: ClassVar[str] = "created_at"
    # Past-tense verb for the output.
    done: ClassVar[str] = "swept"
    # Field values the default `act` writes to stale rows.
    marked: ClassVar[dict[str, Any]] = {}

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--stale-after",
            default=self.default_stale_after,
            help=f"How old a {self.noun} must be to count as stale: 90s, 30m, 2h, 1d "
            f"(default {self.default_stale_after}).",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=self.default_limit,
            help=f"The most {self.noun}s to take in one run (default {self.default_limit}).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help=f"Only count the stale {self.noun}s; change nothing.",
        )

    def stale(self, cutoff: datetime) -> QuerySet[Any]:
        """Return the rows that are stuck since before `cutoff`."""
        raise NotImplementedError

    def act(self, stale: QuerySet[Any], *, limit: int, dry_run: bool) -> int:
        """Handle up to `limit` stale rows and return the count."""
        return sweep(
            stale, limit=limit, order_by=self.order_by, marked=self.marked, dry_run=dry_run
        )

    def handle(self, *args: Any, **options: Any) -> None:
        try:
            age = parse_age(options["stale_after"])
        except ValueError as exc:
            raise CommandError(str(exc)) from exc
        limit = int(options["limit"])
        if limit < 1:
            raise CommandError("--limit must be at least 1")
        dry_run = bool(options["dry_run"])
        count = self.act(self.stale(timezone.now() - age), limit=limit, dry_run=dry_run)
        verb = f"would be {self.done}" if dry_run else self.done
        self.stdout.write(f"{count} {self.noun}{'' if count == 1 else 's'} {verb}")


__all__ = ["SweepCommand", "parse_age", "sweep"]
