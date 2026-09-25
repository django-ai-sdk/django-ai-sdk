from __future__ import annotations

import asyncio

from django_ai_sdk.adapters import base
from django_ai_sdk.adapters.base import Stream
from django_ai_sdk.adapters.citations import (
    CitationRegistry,
    DefaultCitationFormatter,
    NumberedSource,
)
from django_ai_sdk.adapters.citations.parser import normalize_citations
from django_ai_sdk.adapters.citations.streaming import StreamingCitationBuffer
from haystack.dataclasses import StreamingChunk

VALID = {1, 2, 3}


def norm(text: str) -> str:
    return normalize_citations(text, VALID)[0]


def registry_with(*ids: int) -> CitationRegistry:
    registry = CitationRegistry()
    registry.add([NumberedSource(index=i, title=f"doc{i}", content="") for i in ids])
    return registry


class TestNormalizeCitations:
    def test_canonical_tag(self):
        text, cited, _ = normalize_citations('Fact <source id="1" />.', VALID)
        assert text == 'Fact <source id="1" />.'
        assert cited == {1}

    def test_bundled_ids_split(self):
        assert norm('Fact <source id="1, 2" />.') == 'Fact <source id="1" /> <source id="2" />.'

    def test_bracket_fallbacks(self):
        assert norm("Fact [1] and 【2】.") == 'Fact <source id="1" /> and <source id="2" />.'

    def test_invalid_tag_id_dropped(self):
        assert norm('Fact <source id="9" />.') == "Fact ."
        assert norm('Fact <source id="1,9" />.') == 'Fact <source id="1" />.'

    def test_non_citation_brackets_left_alone(self):
        text = "See [9], arr[1], [docs](http://x), [ref][2] and [1]: http://y"
        assert norm(text) == text

    def test_no_citations_inside_code(self):
        text = 'Use `a <source id="2" />[2]` or\n```py\nfoo([1])\n<source id="1" />\n```\nfact [1]'
        out, cited, _ = normalize_citations(text, VALID)
        assert out == 'Use `a [2]` or\n```py\nfoo([1])\n\n```\nfact <source id="1" />'
        assert cited == {1}

    def test_stray_backtick_ends_at_newline(self):
        assert norm("a ` b\nfact [1]") == 'a ` b\nfact <source id="1" />'


class TestStreamingCitationBuffer:
    def test_split_tag_parses_once_complete(self):
        buffer = StreamingCitationBuffer(registry_with(3))
        out = buffer.feed("Fact <sour") + buffer.feed('ce id="3" /> more') + buffer.flush()
        assert out == 'Fact <source id="3" /> more'
        assert buffer.cited == {3}

    def test_code_state_spans_chunks(self):
        buffer = StreamingCitationBuffer(registry_with(1))
        chunks = ["See ``", "`py\nx = [", "1] <sou", 'rce id="1" />\n``', "`\nfact [1]"]
        out = "".join(buffer.feed(c) for c in chunks) + buffer.flush()
        assert out == 'See ```py\nx = [1] \n```\nfact <source id="1" />'

    def test_partial_tag_is_held_back(self):
        buffer = StreamingCitationBuffer(registry_with(1))
        assert buffer.feed("Fact [") == "Fact "
        assert buffer.feed("1] done") == '<source id="1" /> done'

    def test_long_unclosed_text_is_released(self):
        buffer = StreamingCitationBuffer(registry_with(1))
        assert buffer.feed("a < b" + "x" * 80) == "a < b" + "x" * 80

    def test_flush_releases_tail(self):
        buffer = StreamingCitationBuffer(registry_with(1))
        assert buffer.feed("x < y") == "x "
        assert buffer.flush() == "< y"


class TestStreamCitations:
    @staticmethod
    async def _run(registry, *contents):
        queue: asyncio.Queue = asyncio.Queue()
        for content in contents:
            await queue.put(StreamingChunk(content=content))
        await queue.put(base._SENTINEL)
        stream = Stream.__new__(Stream)
        stream.citation_registry = registry
        stream.citations_missing = False
        events = [event async for event in stream.get_events(queue, None)]
        return stream, "".join(e.content for e in events if e.event_type == "text_chunk")

    async def test_stream_normalizes_split_tags(self):
        stream, text = await self._run(registry_with(1), "A <so", 'urce id="1" /> [9]')
        assert text == 'A <source id="1" /> [9]'
        assert stream.cited_ids == {1}
        assert stream.citations_missing is False

    async def test_zero_citations_with_sources_is_flagged(self):
        stream, text = await self._run(registry_with(1, 2), "An answer with no citations.")
        assert text == "An answer with no citations."
        assert stream.citations_missing is True


def test_formatter_reminder_follows_sources():
    text, _ = DefaultCitationFormatter().format([{"content": "x", "meta": {}}], start_index=1)
    assert text.rstrip().endswith(
        'Reminder: cite inline with <source id="N" />, one tag per source id.'
    )
    assert text.index("</source>") < text.index("Reminder:")
