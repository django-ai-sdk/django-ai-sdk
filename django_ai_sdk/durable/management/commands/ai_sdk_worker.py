"""Run the DBOS worker: recover unfinished runs, dequeue workflows and tasks, fire triggers."""

from __future__ import annotations

import logging
import signal
import threading
import time
from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand, CommandError
from django.db import DEFAULT_DB_ALIAS, DatabaseError, close_old_connections, connections

if TYPE_CHECKING:
    from django.db.backends.base.base import BaseDatabaseWrapper

logger = logging.getLogger(__name__)

# How often a standby worker checks whether the running one has stopped.
STANDBY_SECONDS = 2.0
# How often the running worker checks it still holds the lock.
LOCK_CHECK_SECONDS = 5.0
# How often the running worker looks for stale runs (`runs.fail_stale_runs`).
UPKEEP_SECONDS = 60.0


class Command(BaseCommand):
    help = (
        "Launch DBOS in this process: recover the runs a stopped worker left, run queued "
        "workflow runs and tasks, and fire cron triggers. Runs until stopped. One worker "
        "runs at a time; another one started meanwhile waits as a standby."
    )

    # The connection holding the worker lock, on Postgres.
    lock_connection: BaseDatabaseWrapper | None = None

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
        try:
            synced = sync_schedules()
            requeued = requeue_pending()
            worker_launched.send(sender=type(self))
            self.stdout.write(
                f"DBOS worker running: {len(synced)} cron trigger(s), {requeued} pending run(s) "
                "queued again. Stop with Ctrl-C or SIGTERM."
            )
            self.warn_other_versions()
            self.upkeep()
            last_upkeep = time.monotonic()
            while not stop.wait(LOCK_CHECK_SECONDS):
                if not self.holds_worker_lock():
                    raise CommandError(
                        "The worker lost its lock (its database connection dropped). It stops, "
                        "so a restarted worker takes the lock again before it runs anything."
                    )
                if time.monotonic() - last_upkeep >= UPKEEP_SECONDS:
                    self.upkeep()
                    last_upkeep = time.monotonic()
        finally:
            # Let the step in flight finish; DBOS recovers what is left on the next start.
            runtime.destroy()
            if self.lock_connection is not None:
                self.lock_connection.close()

    def warn_other_versions(self) -> None:
        """Warn about unfinished runs this worker's VERSION leaves alone."""
        from django_ai_sdk.durable.conf import setting
        from django_ai_sdk.workflows.durable import runs_under_other_versions

        if others := runs_under_other_versions():
            versions = ", ".join(sorted(set(others.values())))
            message = (
                f"{len(others)} unfinished run(s) wait under AI_SDK_DBOS['VERSION'] {versions}, "
                f"not {setting('VERSION')!r}, and this worker does not run them. Run a worker "
                "of their version until they finish, or cancel and resume them."
            )
            logger.warning(message)
            self.stderr.write(message)

    def upkeep(self) -> None:
        """Fail the runs whose DBOS workflow no longer runs (`runs.fail_stale_runs`)."""
        from django_ai_sdk.workflows.runs import fail_stale_runs

        close_old_connections()
        try:
            if failed := fail_stale_runs():
                self.stderr.write(f"{failed} run(s) had stopped without a worker; failed them.")
        except Exception:
            logger.exception("Could not check for stale workflow runs")

    def take_worker_lock(self, stop: threading.Event) -> bool:
        """Wait until this is the only worker. False when stopped while waiting.

        DBOS recovers every unfinished run at launch, so two workers would both run them.
        A session advisory lock lets a deploy start the new worker first; SQLite: by hand.
        """
        from django_ai_sdk.durable.conf import setting

        if connections[DEFAULT_DB_ALIAS].vendor != "postgresql":
            return True
        key = f"ai_sdk_worker:{setting('APP_NAME')}"
        lock = connections.create_connection(DEFAULT_DB_ALIAS)
        waiting = False
        while not stop.is_set():
            with lock.cursor() as cursor:
                cursor.execute("SELECT pg_try_advisory_lock(hashtext(%s))", [key])
                if cursor.fetchone()[0]:
                    self.lock_connection = lock
                    return True
            if not waiting:
                self.stdout.write("Another worker is running; waiting for it to stop.")
                waiting = True
            stop.wait(STANDBY_SECONDS)
        lock.close()
        return False

    def holds_worker_lock(self) -> bool:
        """Whether the lock's session is still alive and holds it. True without a lock."""
        if self.lock_connection is None:
            return True
        try:
            with self.lock_connection.cursor() as cursor:
                # The session takes no other advisory lock, and a new session holds none.
                cursor.execute(
                    "SELECT count(*) FROM pg_locks "
                    "WHERE locktype = 'advisory' AND granted AND pid = pg_backend_pid()"
                )
                return cursor.fetchone()[0] > 0
        except DatabaseError:
            logger.exception("The worker lock's connection failed")
            return False
