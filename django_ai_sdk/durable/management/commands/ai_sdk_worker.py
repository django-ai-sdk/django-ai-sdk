"""Run the DBOS worker: recover unfinished runs, dequeue workflows and tasks, fire triggers."""

from __future__ import annotations

import signal
import threading
from typing import Any

from django.core.management.base import BaseCommand
from django.db import connection

# How often a standby worker checks whether the running one has stopped.
STANDBY_SECONDS = 2.0


class Command(BaseCommand):
    help = (
        "Launch DBOS in this process: recover the runs a stopped worker left, run queued "
        "workflow runs and tasks, and fire cron triggers. Runs until stopped. One worker "
        "runs at a time; another one started meanwhile waits as a standby."
    )

    def handle(self, *args: Any, **options: Any) -> None:
        from django_ai_sdk.durable import runtime
        from django_ai_sdk.durable.signals import worker_launched
        from django_ai_sdk.workflows.runs import requeue_pending
        from django_ai_sdk.workflows.triggers import sync_schedules

        stop = threading.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: stop.set())

        if not self.take_worker_lock(stop):
            return
        runtime.launch()
        synced = sync_schedules()
        requeued = requeue_pending()
        worker_launched.send(sender=type(self))
        self.stdout.write(
            f"DBOS worker running: {len(synced)} cron trigger(s), {requeued} pending run(s) "
            "queued again. Stop with Ctrl-C or SIGTERM."
        )
        try:
            stop.wait()
        finally:
            # Let the step in flight finish; DBOS recovers what is left on the next start.
            runtime.destroy()

    def take_worker_lock(self, stop: threading.Event) -> bool:
        """Wait until this is the only worker. False when stopped while waiting.

        DBOS recovers every unfinished run of its executor ("local") at launch, so two
        workers at once would both run them. A Postgres advisory lock, held by this
        session until the process exits, lets a deploy start the new worker before
        the old one stops. SQLite is for development: one worker, by hand.
        """
        from django_ai_sdk.durable.conf import setting

        if connection.vendor != "postgresql":
            return True
        key = f"ai_sdk_worker:{setting('APP_NAME')}"
        waiting = False
        with connection.cursor() as cursor:
            while not stop.is_set():
                cursor.execute("SELECT pg_try_advisory_lock(hashtext(%s))", [key])
                if cursor.fetchone()[0]:
                    return True
                if not waiting:
                    self.stdout.write("Another worker is running; waiting for it to stop.")
                    waiting = True
                stop.wait(STANDBY_SECONDS)
        return False
