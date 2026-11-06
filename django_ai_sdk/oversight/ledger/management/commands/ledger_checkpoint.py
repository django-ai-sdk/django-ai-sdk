"""Write a ledger checkpoint, signed when a signing key is set."""

from __future__ import annotations

from typing import Any

from django.core.exceptions import ImproperlyConfigured
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = (
        "Write a checkpoint over every ledger entry chained since the last one, and send "
        "it to the notifiers. Run it on a schedule."
    )

    def handle(self, *args: Any, **options: Any) -> None:
        from django_ai_sdk.oversight.ledger.checkpoints import ChainTruncated, write_checkpoint

        try:
            checkpoint = write_checkpoint()
        except (ImproperlyConfigured, ChainTruncated) as exc:
            raise CommandError(str(exc)) from exc
        if checkpoint is None:
            self.stdout.write("Nothing chained since the last checkpoint.")
            return
        signed = "signed" if checkpoint.signed else "unsigned"
        self.stdout.write(
            self.style.SUCCESS(
                f"Checkpoint at {checkpoint.tree_size} entries, root "
                f"{checkpoint.root_hash[:16]} ({signed})."
            )
        )
