"""Background tasks for oversight, run by a django-tasks worker on the `default` queue.

`enqueue_reflect` and `enqueue_notifications` queue them. A missing task backend
must not fail the turn, so a failure to queue is only logged.
"""

from __future__ import annotations

from typing import Any

from asgiref.sync import async_to_sync
from django_tasks import task

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.notifiers import send_to_notifiers
from django_ai_sdk.oversight.reflection import run_reflection

logger = get_logger(__name__)


@task(queue_name="default")
def reflect_turn(turn_id: str) -> None:
    """Reflect on a finished turn."""
    async_to_sync(run_reflection)(turn_id)


@task(queue_name="default")
def send_notifications(events: list[list[Any]]) -> None:
    """Send a turn's notifications outside the chat stream.

    Each item in `events` is a pair of [event, notifier names].
    """
    for event, names in events:
        async_to_sync(send_to_notifiers)(event, names=names)


def enqueue_notifications(events: list[tuple[dict[str, Any], list[str]]]) -> None:
    """Queue notifications for the worker, so a slow notifier cannot hold the stream open."""
    try:
        send_notifications.enqueue([[event, names] for event, names in events])
    except Exception:
        logger.exception("Could not queue oversight notifications; is a task backend configured?")


def enqueue_reflect(turn_id: Any) -> bool:
    """Queue reflection of `turn_id`. False, and logged, when it could not be queued."""
    try:
        reflect_turn.enqueue(str(turn_id))
    except Exception:
        logger.exception(
            "Could not queue reflection for turn {}; is a task backend configured?", turn_id
        )
        return False
    return True
