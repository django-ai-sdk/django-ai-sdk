"""Forget threads' oversight content as deleting them does, for threads deleted without
signals (a queryset `.update()`, raw SQL, another database). Safe to run again:

    manage.py oversight_forget_thread <thread id> [<thread id> ...]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand

if TYPE_CHECKING:
    from argparse import ArgumentParser


class Command(BaseCommand):
    help = "Forget the oversight content of threads, through the installed custody."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("thread_ids", nargs="+", help="Thread ids.")

    def handle(self, *args: Any, **options: Any) -> None:
        from django_ai_sdk.oversight import custody

        for thread_id in options["thread_ids"]:
            scope = custody.scope_for(thread_id)
            custody.get().forget(scope)
            self.stdout.write(f"Forgot {scope}.")
