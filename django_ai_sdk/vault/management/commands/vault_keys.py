"""Print a new vault master key and the ids of the configured ones."""

from __future__ import annotations

from typing import Any

from django.core.exceptions import ImproperlyConfigured
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Print a new vault master key, and the key ids of the configured ones."

    def handle(self, *args: Any, **options: Any) -> None:
        from django_ai_sdk.vault.keys import generate_master_key, provider

        self.stdout.write(f"new master key: {generate_master_key()}")
        try:
            ids = provider().key_ids()
        except ImproperlyConfigured as exc:
            self.stdout.write(f"configured: none ({exc})")
            return
        self.stdout.write(f"current key id: {ids[0]}")
        for previous in ids[1:]:
            self.stdout.write(f"previous key id: {previous}")
