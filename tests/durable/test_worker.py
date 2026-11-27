from __future__ import annotations

import asyncio
import threading
import time

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
    assert Command().take_worker_lock(stop) is False

    first.close()
    worker = Command()
    assert worker.take_worker_lock(threading.Event()) is True
    worker.lock_connection.close()


@pytest.mark.django_db(transaction=True)
@pytest.mark.usefixtures("postgres_only")
def test_the_lock_is_not_on_django_s_connection_and_its_loss_is_noticed():
    worker = Command()
    assert worker.take_worker_lock(threading.Event()) is True
    connection.close()
    assert worker.holds_worker_lock() is True
    with worker.lock_connection.cursor() as cursor:
        cursor.execute("SELECT pg_backend_pid()")
        [pid] = cursor.fetchone()

    with connection.cursor() as cursor:
        cursor.execute("SELECT pg_terminate_backend(%s)", [pid])

    assert worker.holds_worker_lock() is False
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
    deadline = time.monotonic() + 5
    while len(closes) < 8 and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    assert len(closes) >= 8
