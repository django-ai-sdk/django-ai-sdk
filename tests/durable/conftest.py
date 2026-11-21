"""DBOS in the test process: launched on a SQLite file per test, its steps in its own threads.

The tests that use it need `django_db(transaction=True)`, so DBOS's threads see the rows.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from django_ai_sdk.durable import runtime

STEPS = {
    "echo": "tests.durable.steps.EchoStep",
    "turn": "tests.durable.steps.TurnStep",
    "hold": "tests.durable.steps.HoldStep",
    "quick_approval": "tests.durable.steps.QuickApprovalStep",
}


@pytest.fixture
def dbos(tmp_path, settings):
    from tests.durable import steps

    steps.calls.clear()
    steps.release.clear()
    steps.keys.clear()
    from django.db import connection

    settings.AI_SDK_DBOS = {
        # On Postgres DBOS shares the test database, in a schema of its own.
        **(
            {"SCHEMA": "dbos_test"}
            if connection.vendor == "postgresql"
            else {"DATABASE_URL": f"sqlite:///{tmp_path}/dbos.sqlite3"}
        ),
        "USE_LISTEN_NOTIFY": False,
        "QUEUES": {"workflows": {"polling_interval_sec": 0.1}, "default": {"polling_interval_sec": 0.1}},
    }
    settings.AI_SDK_WORKFLOWS = {"STEPS": STEPS}
    runtime.launch()
    yield runtime
    runtime.destroy()


async def settled(run_id, *, until=("completed", "failed", "cancelled"), step_waiting="", seconds=20):
    """The run once its status is in `until`, or once step `step_waiting` waits."""
    from django_ai_sdk.workflows.models import StepRun, WorkflowRun

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        run = await WorkflowRun.objects.aget(pk=run_id)
        if run.status in until:
            return run
        if step_waiting and await StepRun.objects.filter(
            run=run, step_name=step_waiting, status=StepRun.Status.WAITING
        ).aexists():
            return run
        await asyncio.sleep(0.1)
    raise AssertionError(f"run {run_id} did not settle; it is {run.status}")
