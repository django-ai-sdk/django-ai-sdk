"""For a host's test suite: run queued workflow runs inline instead of on DBOS.

@pytest.fixture(autouse=True)
def _inline_runs(monkeypatch):
    run_inline(monkeypatch)
"""

from __future__ import annotations

import contextlib
from typing import Any


def run_inline(monkeypatch: Any) -> None:
    """Make a queued run execute at once, in the test, with no DBOS schedules to read or write.

    The same `execute_run` the DBOS workflow calls, without checkpoints; a run's
    rows record how it ended, as with a queue. An approval step fails at once,
    since only a run on DBOS waits. `monkeypatch` is pytest's fixture.
    """
    from asgiref.sync import async_to_sync

    from django_ai_sdk.workflows import durable, runs, services, triggers

    def run_now(run_id: str) -> str:
        with contextlib.suppress(Exception):
            async_to_sync(runs.execute_run)(run_id)
        return durable.workflow_id(run_id, 0)

    monkeypatch.setattr(durable, "enqueue", run_now)
    monkeypatch.setattr(triggers, "sync_schedules", lambda only=None: [])

    async def no_schedules() -> dict[str, Any]:
        return {}

    monkeypatch.setattr(services, "_schedules", no_schedules)


__all__ = ["run_inline"]
