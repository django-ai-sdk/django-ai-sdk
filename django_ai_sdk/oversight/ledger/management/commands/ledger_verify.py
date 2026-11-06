"""Verify the ledger chain, its records and the newest checkpoint."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand, CommandError

if TYPE_CHECKING:
    from argparse import ArgumentParser

    from django_ai_sdk.oversight.ledger.schemas import VerifyOut


class Command(BaseCommand):
    help = (
        "Verify the ledger's hash chain and its newest checkpoint. Exits non-zero on a "
        "problem, and when content went unchecked because its master key is not configured "
        "here (unless --allow-unreadable)."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--json", action="store_true", help="Print the result as JSON.")
        parser.add_argument(
            "--verifier",
            default="",
            help="Check checkpoint signatures with this verifier key (name+hash+key), as "
            "printed by ledger_keys.",
        )
        parser.add_argument(
            "--allow-unreadable",
            action="store_true",
            help="Pass when the only finding is content kept under a master key not "
            "configured here.",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        from django_ai_sdk.oversight.ledger import checkpoints
        from django_ai_sdk.oversight.ledger.models import Checkpoint
        from django_ai_sdk.oversight.ledger.services import verify_all

        bad_signatures = 0
        if options["verifier"]:
            try:
                bad_signatures = sum(
                    not checkpoints.signed_by(c, options["verifier"])
                    for c in Checkpoint.objects.filter(signed=True)
                )
            except ValueError as exc:
                raise CommandError("--verifier is not a verifier key (name+hash+key).") from exc
        result = verify_all()
        if bad_signatures:
            result.ok = False
        if options["json"]:
            self.stdout.write(json.dumps({**result.model_dump(), "bad_signatures": bad_signatures}))
        else:
            self._report(result, bad_signatures)
        if not result.ok:
            raise CommandError("The ledger does not verify.")
        if result.unreadable and not options["allow_unreadable"]:
            raise CommandError(
                f"{result.unreadable} content rows are kept under a master key not configured "
                "here and went unchecked."
            )
        if not options["json"]:
            self.stdout.write(self.style.SUCCESS("The ledger verifies."))

    def _report(self, result: VerifyOut, bad_signatures: int) -> None:
        self.stdout.write(
            f"{result.checked} entries checked, {result.pruned} pruned, "
            f"{result.forgotten} forgotten, {result.unreadable} unreadable, "
            f"head {result.head[:16]}"
        )
        if result.checkpoint_size is not None:
            state = "matches" if result.checkpoint_ok else "DOES NOT MATCH"
            self.stdout.write(f"checkpoint at {result.checkpoint_size} {state}")
        if bad_signatures:
            self.stdout.write(self.style.ERROR(f"{bad_signatures} checkpoint(s) badly signed"))
        for unread in result.unread:
            self.stdout.write(self.style.WARNING(f"unreadable: {json.dumps(unread)}"))
        for problem in result.problems:
            self.stdout.write(self.style.ERROR(json.dumps(problem)))
