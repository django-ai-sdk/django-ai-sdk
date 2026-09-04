from __future__ import annotations

import asyncio
import contextvars
from typing import TYPE_CHECKING, Any

from django.core.exceptions import ImproperlyConfigured
from durastream import AsyncStore

from django_ai_sdk.logger import get_logger
from django_ai_sdk.protocols.utils import format_sse
from django_ai_sdk.utils import resolve_setting

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Coroutine

logger = get_logger(__name__)

_store: AsyncStore | None = None

# asyncio.create_task() only keeps a *weak* reference to the task, prevent GC.
_background_tasks: set[asyncio.Task[None]] = set()

_DELETE_GRACE_SECONDS = 2
_LIVE_QUEUE_MAX = 1024

_END = object()


def _report_background_task(task: asyncio.Task[None]) -> None:
    """Retire a finished background task, surfacing any failure."""
    _background_tasks.discard(task)
    if task.cancelled():
        logger.warning("Background stream task was cancelled before completing")
        return
    exc = task.exception()
    if exc is not None:
        logger.opt(exception=exc).error("Background stream task failed")


async def _detached(coro: Coroutine[None, None, None]) -> None:
    """Run `coro` with a thread-sensitive executor of its own."""
    from asgiref.sync import ThreadSensitiveContext

    async with ThreadSensitiveContext():
        await coro


def spawn_background(coro: Coroutine[None, None, None]) -> asyncio.Task[None]:
    # Empty context
    task = asyncio.create_task(_detached(coro), context=contextvars.Context())
    _background_tasks.add(task)
    task.add_done_callback(_report_background_task)
    return task


class StreamSync:
    """In-memory hand-off from the generating task to the HTTP response."""

    __slots__ = ("_queue", "_attached")

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=_LIVE_QUEUE_MAX)
        self._attached = True

    def publish(self, data: bytes) -> None:
        """Forward a chunk to the live reader. Never blocks"""
        if not self._attached:
            return
        try:
            self._queue.put_nowait(data)
        except asyncio.QueueFull:
            logger.debug("Sync relay backed up, detaching.")
            self._attached = False

    def close(self) -> None:
        if self._attached:
            try:
                self._queue.put_nowait(_END)
            except asyncio.QueueFull:
                self._attached = False

    async def stream(self) -> AsyncGenerator[bytes, None]:
        try:
            while True:
                data = await self._queue.get()
                if data is _END:
                    return
                yield data
        finally:
            # Response generator closed.
            self._attached = False


def get_store() -> AsyncStore:
    """Lazy singleton"""
    global _store
    if _store is None:
        path = resolve_setting("AI_SDK_DURABLE_STREAMS_PATH")
        if not path:
            raise ImproperlyConfigured(
                "AI_SDK_DURABLE_STREAMS_PATH must be set to use stream resume."
            )
        _store = AsyncStore(path)
    return _store


async def clear_active_stream(thread_id: str) -> None:
    """Clear the thread's active_stream_id."""
    from django_ai_sdk.storage.base import StorageAdapterRegistry

    try:
        for adapter_class in StorageAdapterRegistry.get_all_adapters():
            if await adapter_class.update_thread(thread_id, None, {"active_stream_id": None}):
                return
    except Exception:
        logger.opt(exception=True).error(
            "Could not clear active_stream_id for thread {}; a later resume may "
            "replay a stale stream",
            thread_id,
        )


async def run_durable_stream(
    sse_gen: AsyncGenerator[bytes, None],
    store: AsyncStore,
    name: str,
    thread_id: str,
    relay: StreamSync | None = None,
) -> None:
    """Detached-task body: drains `sse_gen` to completion regardless of whether
    anything is still listening.
    """
    stream = None
    try:
        stream = await store.create(name, "text/event-stream")
        async for data in sse_gen:
            # Reader first: it should never wait on the disk write.
            if relay is not None:
                relay.publish(data)
            await stream.append(data)
    except Exception:
        logger.opt(exception=True).error("Durable stream {} failed", name)
        raise
    finally:
        # Every step is independently guarded.
        if relay is not None:
            relay.close()
        if stream is not None:
            try:
                await stream.close()
            except Exception:
                logger.opt(exception=True).error("Closing durable stream {} failed", name)

            await clear_active_stream(thread_id)
            try:
                await asyncio.sleep(_DELETE_GRACE_SECONDS)
                await store.delete(name)
            except Exception:
                logger.opt(exception=True).error("Deleting durable stream {} failed", name)


async def tail_stream(store: AsyncStore, name: str) -> AsyncGenerator[bytes, None]:
    """Shared reader for both the live first-attach response and a later resume:"""
    try:
        stream = await store.open(name)
    except KeyError:
        # The generation finished and its log was cleaned up.
        logger.debug("Durable stream {} already cleaned up, nothing to replay", name)
        yield format_sse("[DONE]")
        return

    async for record in stream.subscribe(offset=0):
        yield record
