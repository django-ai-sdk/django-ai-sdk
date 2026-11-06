"""Rewrap every data key under the current master key, so old ones can be dropped."""

from __future__ import annotations

from typing import Any

from django.core.exceptions import ImproperlyConfigured
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = (
        "Rewrap every vault data key under the first AI_SDK_VAULT['MASTER_KEYS'] entry. "
        "Fails when a key opens under none of them."
    )

    def add_arguments(self, parser: Any) -> None:
        from django_ai_sdk.vault.scopes import REWRAP_BATCH

        parser.add_argument("--batch", type=int, default=REWRAP_BATCH, help="Keys per transaction.")

    def handle(self, *args: Any, **options: Any) -> None:
        from django_ai_sdk.vault.scopes import rewrap

        try:
            moved, stuck = rewrap(batch=max(1, options["batch"]))
        except ImproperlyConfigured as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(f"rewrapped {moved} data keys")
        if stuck:
            raise CommandError(
                f"{len(stuck)} data keys open under no configured master key, left as they "
                f"are: {', '.join(stuck[:20])}"
            )
