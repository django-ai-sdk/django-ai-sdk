"""Print the configured verifier key, or new ledger keys when none is set."""

from __future__ import annotations

import secrets
from typing import Any

from django.core.exceptions import ImproperlyConfigured
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = (
        "Print the verifier key of AI_SDK_OVERSIGHT_LEDGER['SIGNING_KEY'] to publish; without one, "
        "print a new SIGNING_KEY, its verifier key and a new PSEUDONYM_SECRET. The vault's "
        "master keys come from `manage.py vault_keys`."
    )

    def handle(self, *args: Any, **options: Any) -> None:
        from django_ai_sdk.oversight.ledger.checkpoints import verifier_key
        from django_ai_sdk.oversight.ledger.conf import origin, setting
        from django_ai_sdk.vault.crypto import new_key_b64

        configured = setting("SIGNING_KEY")
        try:
            if configured:
                self.stdout.write(f"verifier key: {verifier_key(origin(), configured)}")
                return
            key = new_key_b64()
            verifier = verifier_key(origin(), key)
        except ImproperlyConfigured as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(f"SIGNING_KEY      = {key!r}")
        self.stdout.write(f"PSEUDONYM_SECRET = {secrets.token_urlsafe(32)!r}")
        self.stdout.write(f"verifier key: {verifier}")
