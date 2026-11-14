"""Background tasks for oversight, run by a django-tasks worker on the `default` queue."""

from __future__ import annotations

from typing import Any

from asgiref.sync import async_to_sync
from django_tasks import task

from django_ai_sdk.oversight.notifiers import send_to_notifiers
from django_ai_sdk.oversight.reflect import run_reflection


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


__all__ = ["reflect_turn", "send_notifications"]
