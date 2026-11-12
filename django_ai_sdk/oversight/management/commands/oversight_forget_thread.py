"""Forget threads' oversight content as deleting them does, for threads deleted without
signals (a queryset `.update()`, raw SQL, another database). Safe to run again:

    manage.py oversight_forget_thread <thread id> [<thread id> ...]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand
from django.db import transaction

from django_ai_sdk.oversight.content import forget_thread

if TYPE_CHECKING:
    from argparse import ArgumentParser


class Command(BaseCommand):
    help = "Forget the oversight content of threads."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("thread_ids", nargs="+", help="Thread ids.")

    def handle(self, *args: Any, **options: Any) -> None:
        for thread_id in options["thread_ids"]:
            with transaction.atomic():
                forget_thread(thread_id)
            self.stdout.write(f"Forgot thread {thread_id}.")
