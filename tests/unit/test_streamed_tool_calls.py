"""A tool call streamed in fragments is stored and emitted once, with its whole input.

Shapes follow Haystack's generators: both send a first delta with the name and no
arguments, then argument fragments without a name. The Responses API repeats the
call id on every fragment; Chat Completions only sends it on the first.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest
from haystack import Pipeline
from haystack.dataclasses import StreamingChunk, ToolCall, ToolCallDelta, ToolCallResult

from django_ai_sdk.adapters.base import _SENTINEL, Stream
from django_ai_sdk.common import StreamWriter


def delta(**fields):
    return StreamingChunk(content="", index=0, tool_calls=[ToolCallDelta(index=0, **fields)])


def result(call_id):
    origin = ToolCall(tool_name="search", arguments={"q": "hello"}, id=call_id)
    return StreamingChunk(
        content="", index=0, tool_call_result=ToolCallResult(result="hit", origin=origin, error=False)
    )


RESPONSES = [
    delta(id="fc_1", tool_name="search"),
    delta(id="fc_1", arguments='{"q": "'),
    delta(id="fc_1", arguments='hello"}'),
    result("fc_1"),
]
CHAT_COMPLETIONS = [
    delta(id="call_1", tool_name="search", arguments=""),
    delta(arguments='{"q": "'),
    delta(arguments='hello"}'),
    result("call_1"),
]


async def run(chunks):
    queue: asyncio.Queue = asyncio.Queue()
    for chunk in [*chunks, _SENTINEL]:
        queue.put_nowait(chunk)
    stream = Stream(pipeline=Pipeline(), generator=AsyncMock())
    writer = StreamWriter(message_id="m1", storage_callback=None)
    events = [e async for e in stream.get_events(queue, writer)]
    return events, writer.message.tool_calls


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks", [RESPONSES, CHAT_COMPLETIONS], ids=["responses", "chat"])
async def test_a_fragmented_tool_call_is_one_call_with_its_whole_input(chunks):
    events, stored = await run(chunks)

    assert [type(e).__name__ for e in events] == [
        "ToolCallStartEvent",
        "ToolInputCompleteEvent",
        "ToolOutputEvent",
    ]
    assert events[1].tool_input == {"q": "hello"}
    assert [(t["name"], t["arguments"]) for t in stored] == [("search", {"q": "hello"})]
    assert stored[0]["result"]["result"] == "hit"
