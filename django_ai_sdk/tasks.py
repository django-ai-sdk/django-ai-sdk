from __future__ import annotations

from datetime import datetime  # noqa: TC003
from typing import Any

from django_tasks import default_task_backend
from django_tasks.base import TaskResultStatus
from pydantic import BaseModel, Field


class TaskError(BaseModel):
    type: str
    traceback: str


class TaskStatus(BaseModel):
    id: str
    status: str
    enqueued_at: datetime | None
    started_at: datetime | None
    finished_at: datetime | None
    errors: list[TaskError] = Field(default_factory=list)
    return_value: Any | None = None


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
        # `.return_value` raises unless the task actually succeeded — only
        # SUCCESSFUL tasks have one.
        return_value=result.return_value if result.status == TaskResultStatus.SUCCESSFUL else None,
    )
