from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from django_tasks import default_task_backend

if TYPE_CHECKING:
    from datetime import datetime


@dataclass
class TaskError:
    type: str
    traceback: str


@dataclass
class TaskStatus:
    id: str
    status: str
    enqueued_at: datetime | None
    started_at: datetime | None
    finished_at: datetime | None
    errors: list[TaskError] = field(default_factory=list)


async def aget_task_status(task_id: str) -> TaskStatus:
    """Generic task status lookup by django-tasks result ID."""
    result = await default_task_backend.aget_result(task_id)
    return TaskStatus(
        id=str(result.id),
        status=result.status.value,
        enqueued_at=result.enqueued_at,
        started_at=result.started_at,
        finished_at=result.finished_at,
        errors=[
            TaskError(type=e.exception_class_path, traceback=e.traceback)
            for e in (result.errors or [])
        ],
    )
