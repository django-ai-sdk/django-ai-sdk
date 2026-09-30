"""Progress reports from the slow parts of building a chat adapter.

While the adapter is built (tools fetched, knowledge bases opened), the client would
otherwise see one undifferentiated wait. Code on that path calls `report_stage`
or `report_unavailable`; the streaming response forwards them as `data-warmup`
and `data-unavailable` parts. Outside a streamed chat there is no listener and both do nothing.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

_reports: ContextVar[tuple[asyncio.AbstractEventLoop, asyncio.Queue[Any]] | None] = ContextVar(
    "ai_sdk_progress_reports", default=None
)


@contextmanager
def reporting_to(queue: asyncio.Queue[Any]) -> Iterator[None]:
    """Send the reports made by code started inside this block to `queue`.

    A task created inside the block copies the setting, so it keeps reporting after the block.
    """
    token = _reports.set((asyncio.get_running_loop(), queue))
    try:
        yield
    finally:
        _reports.reset(token)


def _send(part: dict[str, Any]) -> None:
    # Thread-safe: a report made inside asyncio.to_thread still reaches the loop in order.
    if (target := _reports.get()) is not None:
        loop, queue = target
        loop.call_soon_threadsafe(queue.put_nowait, part)


def report_stage(stage: str, detail: str = "") -> None:
    """Say which part of the build is running, e.g. `("Engineering documents", "HR documents")`.

    Sent as `data-warmup` with `status: "start"`, so a client that only knows the
    start/ready/failed statuses still shows its wait indicator.
    """
    _send(
        {
            "type": "data-warmup",
            "data": {"status": "start", "stage": stage, "detail": detail},
            "transient": True,
        }
    )


def report_unavailable(stage: str, detail: str = "") -> None:
    """Say a part of the build could not be used, e.g. `("integrations", "Zendesk")`.

    The reply carries on without it.
    """
    _send(
        {
            "type": "data-unavailable",
            "data": {"stage": stage, "detail": detail},
            "transient": True,
        }
    )
