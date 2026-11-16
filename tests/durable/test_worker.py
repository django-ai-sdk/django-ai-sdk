"""The worker command: one at a time, so DBOS never recovers the same runs twice."""

from __future__ import annotations

import threading

import pytest
from django.db import connection, connections

from django_ai_sdk.durable.management.commands.ai_sdk_worker import Command

KEY = "ai_sdk_worker:django-ai-sdk"


@pytest.mark.django_db(transaction=True)
@pytest.mark.usefixtures("postgres_only")
def test_a_second_worker_waits_until_the_first_one_stops():
    first = connections.create_connection("default")
    with first.cursor() as cursor:
        cursor.execute("SELECT pg_advisory_lock(hashtext(%s))", [KEY])

    stop = threading.Event()
    threading.Timer(0.3, stop.set).start()
    # Stopped while it waited as a standby: it never launched.
    assert Command().take_worker_lock(stop) is False

    first.close()  # The first worker's process exits, and its lock with it.
    worker = Command()
    assert worker.take_worker_lock(threading.Event()) is True
    worker.lock_connection.close()


@pytest.mark.django_db(transaction=True)
@pytest.mark.usefixtures("postgres_only")
def test_the_lock_is_not_on_django_s_connection_and_its_loss_is_noticed():
    worker = Command()
    assert worker.take_worker_lock(threading.Event()) is True
    # Django closing its own connection, as close_old_connections does, keeps the lock.
    connection.close()
    assert worker.holds_worker_lock() is True
    with worker.lock_connection.cursor() as cursor:
        cursor.execute("SELECT pg_backend_pid()")
        [pid] = cursor.fetchone()

    with connection.cursor() as cursor:
        # The lock's session drops, as on a database restart or a network blip.
        cursor.execute("SELECT pg_terminate_backend(%s)", [pid])

    assert worker.holds_worker_lock() is False
    # Another worker may take the lock now, so this one stops.
    other = Command()
    assert other.take_worker_lock(threading.Event()) is True
    other.lock_connection.close()


@pytest.mark.django_db(transaction=True)
async def test_every_durable_unit_runs_between_closes_of_stale_connections(dbos, monkeypatch):
    from django_ai_sdk.durable import runtime
    from django_ai_sdk.workflows.runs import queue_run
    from django_ai_sdk.workflows.schemas import StepDefinition, WorkflowDefinition
    from tests.durable.conftest import settled

    closes = []
    monkeypatch.setattr(runtime, "close_old_connections", lambda: closes.append(1))
    run = await queue_run(WorkflowDefinition(steps=[StepDefinition(type="echo", name="a")]))

    assert (await settled(run.pk)).status == "completed"
    # Before and after each of claim, run:start, step:a and run:end.
    assert len(closes) >= 8
