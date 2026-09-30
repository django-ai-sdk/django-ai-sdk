"""A chat says which part of the adapter build it waits on, and which parts were skipped."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from django_ai_sdk.events import MessageStartEvent, StreamEndEvent, StreamEvent
from django_ai_sdk.protocols.vercel import VercelProtocolHandler
from django_ai_sdk.progress import report_stage, report_unavailable
from django_ai_sdk.responses import stream_response


class _Adapter:
    model: str | None = None
    instructions: str | None = None
    suggestion_generator = None

    async def stream(self, messages: list[Any]) -> AsyncGenerator[StreamEvent, None]:
        yield MessageStartEvent(message_id="m1")
        yield StreamEndEvent()


async def parts(build: Any) -> list[dict[str, Any]]:
    """The data-warmup and data-unavailable parts a chat streams, in order."""
    response = await stream_response(build, [], VercelProtocolHandler())
    text = b"".join([chunk async for chunk in response.streaming_content]).decode()
    found = []
    for line in text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            part = json.loads(line.removeprefix("data: "))
            if part.get("type") in ("data-warmup", "data-unavailable"):
                found.append(part)
    return found


def statuses(found: list[dict[str, Any]]) -> list[str]:
    return [p["data"]["status"] for p in found if p["type"] == "data-warmup"]


@pytest.mark.asyncio
async def test_stages_arrive_between_start_and_ready() -> None:
    async def build() -> _Adapter:
        report_stage("integrations", "MediaWiki")
        await asyncio.sleep(0)
        report_stage("knowledge", "HR Knowledge")
        return _Adapter()

    found = await parts(build)

    assert [
        (p["data"]["status"], p["data"].get("stage"), p["data"].get("detail")) for p in found
    ] == [
        ("start", None, None),
        ("start", "integrations", "MediaWiki"),
        ("start", "knowledge", "HR Knowledge"),
        ("ready", None, None),
    ]


@pytest.mark.asyncio
async def test_a_client_that_only_knows_start_ready_failed_still_sees_a_wait() -> None:
    async def build() -> _Adapter:
        report_stage("knowledge", "HR Knowledge")
        return _Adapter()

    assert set(statuses(await parts(build))) <= {"start", "ready", "failed"}


@pytest.mark.asyncio
async def test_an_unavailable_part_is_streamed_before_ready() -> None:
    async def build() -> _Adapter:
        report_unavailable("integrations", "MediaWiki")
        return _Adapter()

    found = await parts(build)

    assert found[1] == {
        "type": "data-unavailable",
        "data": {"stage": "integrations", "detail": "MediaWiki"},
        "transient": True,
    }
    assert statuses(found)[-1] == "ready"


@pytest.mark.asyncio
async def test_a_failed_build_still_ends_in_failed_and_never_ready() -> None:
    async def build() -> _Adapter:
        report_stage("knowledge", "HR Knowledge")
        raise RuntimeError("index locked")

    found = await parts(build)

    assert statuses(found) == ["start", "start", "failed"]


@pytest.mark.asyncio
async def test_a_report_made_in_a_worker_thread_is_delivered() -> None:
    async def build() -> _Adapter:
        await asyncio.to_thread(report_stage, "knowledge", "from a thread")
        return _Adapter()

    found = await parts(build)

    assert [p["data"].get("detail") for p in found] == [None, "from a thread", None]


@pytest.mark.asyncio
async def test_reports_outside_a_streamed_chat_do_nothing() -> None:
    report_stage("knowledge", "HR Knowledge")
    report_unavailable("integrations", "MediaWiki")


@pytest.mark.asyncio
async def test_a_prebuilt_adapter_streams_no_stages() -> None:
    response = await stream_response(_Adapter(), [], VercelProtocolHandler())
    text = b"".join([chunk async for chunk in response.streaming_content]).decode()

    assert '"stage"' not in text
