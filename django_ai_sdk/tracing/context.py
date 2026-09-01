from __future__ import annotations

import asyncio
import contextlib
import contextvars
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Coroutine, Generator

    from haystack.tracing import Span

from django_ai_sdk.logger import logger

_correlation: contextvars.ContextVar[dict[str, uuid.UUID] | None] = contextvars.ContextVar(
    "django_ai_sdk_tracing_correlation", default=None
)
_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "django_ai_sdk_tracing_current_span", default=None
)

_pending_writes: set[asyncio.Task[Any]] = set()


def current() -> dict[str, uuid.UUID] | None:
    """Return the active correlation ids, if any."""
    return _correlation.get()


def _coerce_id(value: str | uuid.UUID | None) -> uuid.UUID | None:
    if isinstance(value, uuid.UUID):
        return value
    if isinstance(value, str):
        with contextlib.suppress(ValueError):
            return uuid.UUID(value)
    return None


@contextlib.contextmanager
def bind(
    thread_id: str | uuid.UUID | None = None, message_id: str | uuid.UUID | None = None
) -> Generator[None]:
    """Correlate every trace span created inside this context with a thread and message."""
    merged = dict(_correlation.get() or {})
    for key, value in (("thread_id", thread_id), ("message_id", message_id)):
        if (coerced := _coerce_id(value)) is not None:
            merged[key] = coerced
    token = _correlation.set(merged or None)
    try:
        yield
    finally:
        _correlation.reset(token)


@contextlib.contextmanager
def active_span(span: Span) -> Generator[None]:
    """Make ``span`` the span tracer sees"""
    token = _current_span.set(span)
    try:
        yield
    finally:
        _current_span.reset(token)


def current_span() -> Span | None:
    """Return the innermost span of the context"""
    return _current_span.get()


def on_event_loop() -> bool:
    """Whether the calling thread is running inside an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def schedule(write: Coroutine[Any, Any, Any]) -> None:
    """Run a trace write on the running loop"""
    task = asyncio.get_running_loop().create_task(write)
    _pending_writes.add(task)
    task.add_done_callback(_pending_writes.discard)
    task.add_done_callback(_log_write_failure)


def _log_write_failure(task: asyncio.Task[Any]) -> None:
    if not task.cancelled() and (exc := task.exception()):
        logger.error("Trace write failed: {}", exc, exc_info=exc)


async def aflush() -> None:
    """Wait until every trace write scheduled"""
    loop = asyncio.get_running_loop()
    while current := {task for task in _pending_writes if task.get_loop() is loop}:
        await asyncio.wait(current)
