from __future__ import annotations

import logging
from datetime import datetime  # noqa: TC003
from typing import TYPE_CHECKING, Any

from django.core.exceptions import ObjectDoesNotExist
from django_tasks import default_task_backend
from django_tasks.base import TaskResultStatus
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser

logger = logging.getLogger(__name__)


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


async def aget_principal(user_id: Any, *, source: str = "") -> AbstractBaseUser | None:
    """Load the user a queued run was created for, or None if it has no user.

    A worker is a fresh process with no request, so per-user integration credentials
    resolve to nothing without this. A database failure propagates: finishing under the
    wrong principal is worse than retrying.
    """
    if user_id is None:
        return None

    from django.contrib.auth import get_user_model

    try:
        return await get_user_model().objects.aget(pk=user_id)
    except (ObjectDoesNotExist, ValueError, TypeError):
        logger.warning(
            "%s targets user %s, who no longer exists", source or "A queued run", user_id
        )
        return None
