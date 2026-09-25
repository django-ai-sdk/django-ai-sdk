"""One agent, one handler, many concurrent streams: no stream may see another's state."""

import json

import pytest

from django_ai_sdk.events import (
    MessageEndEvent,
    MessageStartEvent,
    StreamEndEvent,
    TextChunkEvent,
    ToolCallStartEvent,
    ToolInputCompleteEvent,
)
from django_ai_sdk.protocols.openai import OpenAIProtocolHandler
from django_ai_sdk.protocols.vercel import VercelProtocolHandler


class Scripted:
    """An adapter that streams a fixed list of events."""

    def __init__(self, *events):
        self.events = events

    async def stream(self, messages):
        for event in self.events:
            yield event


def reply(message_id, *texts):
    return Scripted(
        MessageStartEvent(message_id=message_id),
        *(TextChunkEvent(content=text) for text in texts),
        MessageEndEvent(finish_reason="stop"),
        StreamEndEvent(),
    )


def parse(raw):
    body = raw.decode().removeprefix("data: ").strip()
    return body if body == "[DONE]" else json.loads(body)


async def drain(stream):
    return [parse(chunk) async for chunk in stream]


@pytest.mark.asyncio
async def test_interleaved_vercel_streams_keep_their_own_text_ids():
    handler = VercelProtocolHandler()
    first = handler.sse(reply("m1", "a1", "a2"), [])

    # The first stream is mid-text when a second request starts and finishes.
    head = [parse(await anext(first)) for _ in range(3)]  # start, text-start, delta
    second = await drain(handler.sse(reply("m2", "b1"), []))
    first_chunks = head + await drain(first)

    for chunks in (first_chunks, second):
        text = [c for c in chunks if isinstance(c, dict) and c["type"].startswith("text-")]
        assert [c["type"] for c in text][0] == "text-start"
        assert len({c["id"] for c in text}) == 1, text


@pytest.mark.asyncio
async def test_an_openai_stream_does_not_repeat_the_previous_streams_tool_calls():
    handler = OpenAIProtocolHandler()
    with_tool = Scripted(
        ToolCallStartEvent(tool_call_id="call-1", tool_name="search"),
        ToolInputCompleteEvent(tool_call_id="call-1", tool_name="search", tool_input={"q": "x"}),
        MessageEndEvent(finish_reason="tool_calls"),
    )
    await drain(handler.sse(with_tool, []))

    chunks = await drain(handler.sse(reply("m2", "plain"), []))

    final = next(c for c in chunks if isinstance(c, dict) and c["choices"][0]["finish_reason"])
    assert not final["choices"][0]["delta"].get("tool_calls")
