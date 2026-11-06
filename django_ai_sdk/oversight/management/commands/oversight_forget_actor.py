"""Forget users on the oversight record as deleting them does, for a delete whose forgetting
failed and was logged. Safe to run again:

    manage.py oversight_forget_actor <user id> [<user id> ...]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand
from django.db import transaction

from django_ai_sdk.oversight.signals import forget_actor

if TYPE_CHECKING:
    from argparse import ArgumentParser


class Command(BaseCommand):
    help = "Forget deleted users on the oversight record: who asked, and threadless content."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("actor_ids", nargs="+", help="User primary keys.")

    def handle(self, *args: Any, **options: Any) -> None:
        for actor_id in options["actor_ids"]:
            with transaction.atomic():
                turns = forget_actor(str(actor_id))
            self.stdout.write(f"Forgot user {actor_id} on {turns} turns.")
