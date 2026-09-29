"""No exception text may reach a client: not live, not on reload, in any protocol."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from haystack import Pipeline

from django_ai_sdk.adapters.base import Stream
from django_ai_sdk.common import ChatMessage
from django_ai_sdk.events import ErrorEvent
from django_ai_sdk.protocols.openai import OpenAIProtocolHandler
from django_ai_sdk.protocols.vercel import VercelProtocolHandler
from django_ai_sdk.responses import _FailedStream, stream_response
from django_ai_sdk.storage.memory import MemoryStorageAdapter

SECRET = "/var/www/SECRET/stores/qdrant is already accessed"
PROTOCOLS = [VercelProtocolHandler, OpenAIProtocolHandler]


async def body(response) -> str:
    return b"".join([chunk async for chunk in response.streaming_content]).decode()


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", PROTOCOLS)
async def test_a_failed_adapter_build_streams_a_code_and_stores_the_text(protocol):
    storage = MemoryStorageAdapter(await MemoryStorageAdapter.create_thread(title="t"))

    async def build():
        raise RuntimeError(SECRET)

    text = await body(await stream_response(build, [], protocol(), storage_adapter=storage))

    assert "SECRET" not in text
    assert "unknown" in text
    [stored] = await storage.get_messages()
    assert stored.error_code == "unknown"
    assert stored.error_ref and stored.error_ref in text
    assert stored.errors == [SECRET]


@pytest.mark.asyncio
async def test_an_error_without_text_stores_its_type():
    storage = MemoryStorageAdapter(await MemoryStorageAdapter.create_thread(title="t"))

    async def build():
        raise TimeoutError

    text = await body(await stream_response(build, [], VercelProtocolHandler(), storage_adapter=storage))

    assert "provider_timeout" in text
    [stored] = await storage.get_messages()
    assert stored.errors == ["TimeoutError"]


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", PROTOCOLS)
async def test_a_failed_pipeline_streams_a_code(protocol):
    stream = Stream(pipeline=Pipeline(), generator=MagicMock(spec=[]))

    with patch.object(Stream, "get_task", side_effect=RuntimeError(SECRET)):
        text = "".join([c.decode() async for c in protocol().sse(stream, [])])

    assert "SECRET" not in text
    assert "unknown" in text


def test_a_reloaded_failure_shows_codes_only():
    message = ChatMessage(
        role="assistant",
        id="m1",
        errors=[SECRET],
        error_code="knowledge_unavailable",
        error_ref="abcd1234",
        tool_calls=[
            {
                "id": "t1",
                "name": "search",
                "arguments": {"q": "x"},
                "result": {"result": SECRET, "origin": {}, "error": True},
            },
            {"id": "t2", "name": "today", "arguments": {}, "result": {"result": "2026-09-28"}},
        ],
    )

    [replayed] = VercelProtocolHandler().from_chat_messages([message])

    assert "SECRET" not in json.dumps(replayed)
    failed, ok = (p for p in replayed["parts"] if p["type"].startswith("tool-"))
    assert failed["state"] == "output-error" and failed["errorText"] == "tool_failed"
    assert ok["state"] == "output-available" and ok["output"] == {"result": "2026-09-28"}
    [error] = [p for p in replayed["parts"] if p["type"] == "data-error"]
    assert error["data"] == {
        "code": "knowledge_unavailable",
        "message": "The knowledge base is temporarily unavailable. Please try again in a moment.",
        "retryable": True,
        "ref": "abcd1234",
    }
    assert replayed["tool_calls"][0]["result"] == {"error": "tool_failed"}
    assert replayed["tool_calls"][1]["result"] == {"result": "2026-09-28"}


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", PROTOCOLS)
async def test_an_unregistered_code_streams_as_unknown(protocol):
    stream = _FailedStream(ErrorEvent(error_code="made_up", ref="abcd1234"), "m1")

    text = "".join([c.decode() async for c in protocol().sse(stream, [])])

    assert "made_up" not in text
    assert "abcd1234" in text


def test_a_reloaded_unregistered_code_is_unknown():
    message = ChatMessage(role="assistant", id="m1", errors=["x"], error_code="made_up", error_ref="r1")

    [replayed] = VercelProtocolHandler().from_chat_messages([message])

    [error] = [p for p in replayed["parts"] if p["type"] == "data-error"]
    assert error["data"] == {
        "code": "unknown",
        "message": "Something went wrong. Please try again.",
        "retryable": True,
        "ref": "r1",
    }
