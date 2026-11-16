"""django-tasks on DBOS: one queue system for workflows and every other task.

    TASKS = {"default": {"BACKEND": "django_ai_sdk.durable.tasks.DBOSTaskBackend"}}

A task runs as a DBOS workflow of one step, on the queue its `queue_name` names
(AI_SDK_DBOS['QUEUES'] declares them). A worker that dies mid-task runs it again
once DBOS recovers it, so a task must be safe to repeat, as with any queue.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from inspect import iscoroutinefunction
from typing import TYPE_CHECKING, Any

from asgiref.sync import async_to_sync, sync_to_async
from dbos import DBOS
from django.db import close_old_connections
from django.utils import timezone
from django.utils.module_loading import import_string
from django_tasks.backends.base import BaseTaskBackend
from django_tasks.base import Task, TaskError, TaskResult, TaskResultStatus
from django_tasks.signals import task_enqueued
from django_tasks.utils import get_random_id, normalize_json

from django_ai_sdk.durable import runtime

if TYPE_CHECKING:
    from dbos import EnqueueOptions, WorkflowStatus

# The DBOS workflow every task runs as.
TASK_WORKFLOW = "ai_sdk.task"

_STATUSES = {
    "ENQUEUED": TaskResultStatus.READY,
    "DELAYED": TaskResultStatus.READY,
    "PENDING": TaskResultStatus.RUNNING,
    "SUSPENDED": TaskResultStatus.RUNNING,
    "SUCCESS": TaskResultStatus.SUCCESSFUL,
}


@DBOS.workflow(name=TASK_WORKFLOW)
async def run_task(module_path: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
    return await DBOS.run_step_async({"name": module_path}, _call, module_path, args, kwargs)


async def _call(module_path: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
    task: Task = import_string(module_path)
    if iscoroutinefunction(task.func):
        async with runtime.fresh_connections():
            value = await task.func(*args, **kwargs)
    else:
        # Off the event loop, and not serialized behind other tasks' sync code.
        value = await sync_to_async(_in_fresh_connections, thread_sensitive=False)(
            task.func, args, kwargs
        )
    return normalize_json(value)


def _in_fresh_connections(func: Any, args: list[Any], kwargs: dict[str, Any]) -> Any:
    """`func` between closes of this thread's broken or expired connections."""
    close_old_connections()
    try:
        return func(*args, **kwargs)
    finally:
        close_old_connections()


class DBOSTaskBackend(BaseTaskBackend):
    supports_async_task = True
    supports_get_result = True
    supports_defer = True

    def validate_task(self, task: Task) -> None:
        super().validate_task(task)
        if task.takes_context:
            from django_tasks.exceptions import InvalidTask

            raise InvalidTask("DBOSTaskBackend does not pass a TaskContext.")

    def enqueue(self, task: Task, args: Any, kwargs: Any) -> TaskResult:
        return async_to_sync(self.aenqueue)(task, args, kwargs)

    async def aenqueue(self, task: Task, args: Any, kwargs: Any) -> TaskResult:
        self.validate_task(task)
        result = TaskResult(
            task=task,
            id=get_random_id(),
            status=TaskResultStatus.READY,
            enqueued_at=timezone.now(),
            started_at=None,
            last_attempted_at=None,
            finished_at=None,
            args=args,
            kwargs=kwargs,
            backend=self.alias,
            errors=[],
            worker_ids=[],
        )
        options: EnqueueOptions = {
            "queue_name": task.queue_name,
            "workflow_name": TASK_WORKFLOW,
            "workflow_id": result.id,
            "app_version": runtime.setting("VERSION"),
        }
        if task.run_after is not None:
            options["delay_seconds"] = max(0.0, (task.run_after - timezone.now()).total_seconds())
        await runtime.client().enqueue_async(options, task.module_path, result.args, result.kwargs)
        task_enqueued.send(type(self), task_result=result)
        return result

    def get_result(self, result_id: str) -> TaskResult:
        return async_to_sync(self.aget_result)(result_id)

    async def aget_result(self, result_id: str) -> TaskResult:
        from django_tasks.exceptions import TaskResultDoesNotExist

        try:
            status = await runtime.status(result_id)
        except Exception as exc:
            raise TaskResultDoesNotExist(result_id) from exc
        if status is None or status.name != TASK_WORKFLOW:
            raise TaskResultDoesNotExist(result_id)
        return self._result(status)

    def _result(self, status: WorkflowStatus) -> TaskResult:
        from django_tasks.exceptions import TaskResultDoesNotExist

        if status.input is None:
            raise TaskResultDoesNotExist(status.workflow_id)
        # run_task(module_path, args, kwargs), as aenqueue queued it.
        module_path, args, kwargs = status.input["args"]
        state = _STATUSES.get(str(status.status), TaskResultStatus.FAILED)
        finished = _moment(status.updated_at) if state in _DONE else None
        result = TaskResult(
            task=import_string(module_path),
            id=status.workflow_id,
            status=state,
            enqueued_at=_moment(status.created_at),
            started_at=_moment(status.dequeued_at),
            last_attempted_at=_moment(status.dequeued_at),
            finished_at=finished,
            args=args,
            kwargs=kwargs,
            backend=self.alias,
            errors=[
                TaskError(
                    exception_class_path=_error_path(status.error), traceback=str(status.error)
                )
            ]
            if status.error is not None
            else [],
            worker_ids=[status.executor_id] if status.executor_id else [],
        )
        if state == TaskResultStatus.SUCCESSFUL:
            object.__setattr__(result, "_return_value", status.output)
        return result


_DONE = {TaskResultStatus.SUCCESSFUL, TaskResultStatus.FAILED}


def _moment(epoch_ms: int | None) -> datetime | None:
    if not epoch_ms:
        return None
    return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=epoch_ms)


def _error_path(error: Exception) -> str:
    # A stored error comes back as PortableWorkflowError, carrying the original's name.
    name = getattr(error, "name", "") or type(error).__name__
    return name if "." in name else f"builtins.{name}"


__all__ = ["DBOSTaskBackend", "TASK_WORKFLOW", "run_task"]
