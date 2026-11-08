"""Delete expired oversight rows (`oversight.retention.prune`). Schedule it daily:

manage.py oversight_prune [--batch-size 500]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand

from django_ai_sdk.oversight.retention import BATCH_SIZE, prune

if TYPE_CHECKING:
    from argparse import ArgumentParser


class Command(BaseCommand):
    help = "Delete expired oversight content, turns and access records."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--batch-size",
            type=int,
            default=BATCH_SIZE,
            help=f"Rows deleted per transaction (default {BATCH_SIZE}).",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        counts = prune(batch_size=options["batch_size"])
        self.stdout.write(
            f"Deleted {counts['content']} content rows, {counts['turns']} turns "
            f"and {counts['access']} access records."
        )
        if counts["open_reviews"]:
            self.stdout.write(f"{counts['open_reviews']} reviews were still open on deleted turns.")
