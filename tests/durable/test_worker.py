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
    assert Command().take_worker_lock(threading.Event()) is True
    with connection.cursor() as cursor:
        cursor.execute("SELECT pg_advisory_unlock_all()")
