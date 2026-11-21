"""DBOS in this process: launched in the worker, a client everywhere else.

Only the worker (`manage.py ai_sdk_worker`) launches DBOS, which recovers
unfinished runs, dequeues and fires schedules. Web processes enqueue, send and
cancel through a `DBOSClient`, which reads and writes the same tables without
running anything.
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from importlib import import_module
from typing import TYPE_CHECKING, Any

from asgiref.sync import sync_to_async
from django.db import close_old_connections

from django_ai_sdk.durable.conf import database_url, setting

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from dbos import DBOSClient, DBOSConfig

logger = logging.getLogger(__name__)

# The modules that declare DBOS workflows. A workflow must be registered before
# DBOS launches, or its unfinished runs are not recovered.
WORKFLOW_MODULES = (
    "django_ai_sdk.durable.tasks",
    "django_ai_sdk.workflows.durable",
)

_client: DBOSClient | None = None
_lock = threading.Lock()


def serializer() -> Any:
    """JSON, never pickle: what is stored can be read, and loading it runs no code."""
    from dbos import DBOSPortableJSONSerializer

    return DBOSPortableJSONSerializer()


def config() -> DBOSConfig:
    return {
        "name": setting("APP_NAME"),
        "system_database_url": database_url(),
        "dbos_system_schema": setting("SCHEMA"),
        "application_version": setting("VERSION"),
        "serializer": serializer(),
        "use_listen_notify": setting("USE_LISTEN_NOTIFY"),
    }


@asynccontextmanager
async def fresh_connections() -> AsyncGenerator[None]:
    """Close broken or expired database connections around a durable unit.

    What Django does around a request, so a database restart fails the unit in
    flight, not every unit after it. Async ORM calls share one thread, and so do
    these closes.
    """
    await sync_to_async(close_old_connections)()
    try:
        yield
    finally:
        await sync_to_async(close_old_connections)()


def register_workflows() -> None:
    for module in WORKFLOW_MODULES:
        import_module(module)


def launch() -> None:
    """Start DBOS here: recover unfinished runs, dequeue, fire schedules. The worker's job."""
    from dbos import DBOS

    register_workflows()
    DBOS(config=config())
    DBOS.launch()
    for name, options in setting("QUEUES").items():
        DBOS.register_queue(name, **options)


def destroy() -> None:
    """Stop DBOS here and drop the client, such as between tests."""
    global _client
    from dbos import DBOS

    DBOS.destroy()
    with _lock:
        if _client is not None:
            _client.destroy()
        _client = None


def client() -> DBOSClient:
    """The process's DBOSClient, made on first use."""
    global _client
    with _lock:
        if _client is None:
            from dbos import DBOSClient

            _client = DBOSClient(
                system_database_url=database_url(),
                dbos_system_schema=setting("SCHEMA"),
                serializer=serializer(),
                application_name=setting("APP_NAME"),
            )
        return _client


async def enqueue(workflow: str, *args: Any, workflow_id: str, queue: str) -> str:
    """Queue a run of the DBOS workflow named `workflow`. Returns its workflow id.

    The id is the idempotency key: queueing an id that exists returns it and runs
    nothing.
    """
    handle = await client().enqueue_async(
        {
            "queue_name": queue,
            "workflow_name": workflow,
            "workflow_id": workflow_id,
            "app_version": setting("VERSION"),
        },
        *args,
    )
    return handle.get_workflow_id()


async def send(workflow_id: str, message: Any, *, topic: str, idempotency_key: str) -> None:
    """Deliver `message` to a run waiting on `topic`. A repeated key is delivered once."""
    await client().send_async(workflow_id, message, topic, idempotency_key)


async def cancel(workflow_id: str) -> None:
    """Stop a run at its next step. The step in flight finishes."""
    await client().cancel_workflow_async(workflow_id)


async def status(workflow_id: str) -> Any:
    """DBOS's WorkflowStatus for `workflow_id`, or None."""
    if not workflow_id:
        return None
    handle = await client().retrieve_workflow_async(workflow_id)
    return await handle.get_status()


__all__ = [
    "cancel",
    "client",
    "enqueue",
    "launch",
    "send",
    "status",
]
