"""Oversight's django-tasks. A missing backend must not fail the turn, so queueing only logs."""

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
    async_to_sync(run_reflection)(turn_id)


@task(queue_name="default")
def send_notifications(events: list[list[Any]]) -> None:
    """Each item in `events` is [event, notifier names]."""
    for event, names in events:
        async_to_sync(send_to_notifiers)(event, names=names)


def enqueue_notifications(events: list[tuple[dict[str, Any], list[str]]]) -> None:
    try:
        send_notifications.enqueue([[event, names] for event, names in events])
    except Exception:
        logger.exception("Could not queue oversight notifications; is a task backend configured?")


def enqueue_reflect(turn_id: Any) -> bool:
    """False, and logged, when it could not be queued."""
    try:
        reflect_turn.enqueue(str(turn_id))
    except Exception:
        logger.exception(
            "Could not queue reflection for turn {}; is a task backend configured?", turn_id
        )
        return False
    return True
