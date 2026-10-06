"""Grounded citations: sources of a turn, the citation agent's reply, and where its quotes are."""

from __future__ import annotations

import asyncio
import json

import pytest
from django_ai_sdk.adapters.base import _SENTINEL, Stream
from django_ai_sdk.adapters.citations import (
    NumberedSource,
    SourceRegistry,
    SourcesFormatter,
    collect_sources,
    from_document,
    from_web_result,
    source_key,
)
from django_ai_sdk.adapters.citations.grounding import ground, locate, parse_pairs
from django_ai_sdk.common import StreamWriter
from haystack.dataclasses import Document
from haystack.tools import Tool

ANSWER = (
    "The **Postal Address** element is optional (0..*). "
    "It is not required for customers or suppliers."
)
CHUNK = Document(
    id="c1",
    content="## POSTAL ADDRESS 0..*, F  xml tag: postalAddress",
    meta={"doc_id": "d1", "file_name": "XAF_4.0.pdf", "split_id": 3},
)


class CitationAgent:
    """Answers like a citation agent would: the JSON `reply`."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[str] = []

    async def run(self, messages, system_prompt=None, response_format=None):
        self.calls.append(messages[0].content)
        return self.reply


def citations(*pairs: tuple[str, list[tuple[int, str]]]) -> str:
    """A citation agent's reply; support as `[(n, "passage"), …]`."""
    return json.dumps({"citations": [
        {"quote": q, "support": [{"source": n, "text": t} for n, t in s]} for q, s in pairs
    ]})


class TestSources:
    def test_a_document_is_titled_by_file_and_section(self):
        source = from_document(CHUNK)
        assert source.title == "XAF_4.0.pdf · §4"
        assert source_key(source) == "doc:d1:c1"

    def test_a_web_result_is_keyed_by_url(self):
        source = from_web_result({"title": "XAF", "url": "https://x.nl/xaf", "content": "..."})
        assert (source.kind, source_key(source)) == ("web", "url:https://x.nl/xaf")

    def test_the_same_source_found_twice_keeps_one_number(self):
        registry = SourceRegistry()
        first = registry.add([from_document(CHUNK)])
        again = registry.add([from_document(CHUNK), from_web_result({"url": "https://x.nl"})])
        assert [s.index for s in first + again] == [1, 1, 2]
        assert len(registry) == 2

    def test_a_seeded_source_without_a_key_gets_one_and_is_stored_only_when_cited(self):
        registry = SourceRegistry()
        registry.seed([NumberedSource(title="cv.pdf", content="...", doc_id="e1", chunk_id="file")])
        assert [s.key for s in registry.all] == ["doc:e1:file"]
        assert registry.seeded == {"doc:e1:file"}

    def test_the_model_reads_sources_without_numbers_or_rules(self):
        text = SourcesFormatter().render([from_document(CHUNK)])
        assert text.startswith("### XAF_4.0.pdf · §4\n## POSTAL ADDRESS")
        assert "source id" not in text and "[1]" not in text

    def test_a_tool_registers_what_it_returns(self):
        registry = SourceRegistry()
        tool = Tool(name="search_web", description="", parameters={}, function=lambda: [])
        collect_sources(tool, registry, key=None, to_source=from_web_result)

        text = tool.outputs_to_string["handler"]([{"title": "XAF", "url": "https://x.nl"}])

        assert "### XAF (https://x.nl)" in text
        assert [s.key for s in registry.all] == ["url:https://x.nl"]


class TestLocate:
    def test_exact(self):
        start = ANSWER.index("It is not required")
        assert locate(ANSWER, "It is not required") == (start, start + len("It is not required"))

    def test_markdown_and_whitespace_are_ignored(self):
        start, end = locate(ANSWER, "The Postal  Address element is optional")
        assert ANSWER[start:end] == "The **Postal Address** element is optional"

    def test_a_copying_slip_is_tolerated(self):
        start, end = locate(ANSWER, "It is not requird for customers or supliers.")
        assert ANSWER[start:end] == "It is not required for customers or suppliers."

    def test_typographic_dashes_and_quotes_match_plain_ones(self):
        text = 'The XML-Auditfile says "optional".'
        assert locate(text, "The XML\u2011Auditfile says \u201coptional\u201d.") == (0, len(text))

    def test_a_repeated_phrase_maps_after_the_previous_citation(self):
        text = "It is optional. Here it is optional too."
        assert locate(text, "optional", start=14) == (27, 35)

    def test_table_rows_copied_without_their_pipes_are_found(self):
        table = "|Element|Card.|\n|---|---|\n|Postal Address|0..*, O|\n|Street|0..1|"
        start, end = locate(table, "Postal Address 0..*, O\nStreet 0..1")
        assert table[start:end] == "Postal Address|0..*, O|\n|Street|0..1"

    def test_unrelated_text_is_not_found(self):
        assert locate(ANSWER, "The invoice date is mandatory for purchase invoices") is None


class TestParsePairs:
    def test_support_passages_per_source(self):
        reply = '{"citations": [{"quote": "a", "support": [{"source": 1, "text": "x y"}, {"source": "?"}]}]}'
        assert parse_pairs(reply) == [("a", [(1, "x y")])]

    def test_a_json_wrapper_is_handled_and_a_reply_without_passages_gives_nothing(self):
        reply = '```json\n{"citations": [{"quote": "a", "support": [{"source": 1, "text": "x"}]}, {"quote": "b", "sources": [1]}]}\n```'
        assert parse_pairs(reply) == [("a", [(1, "x")])]

    def test_text_around_the_json_is_ignored(self):
        reply = 'Sure: {"citations": [{"quote": "b", "support": [{"source": 2, "text": "y"}]}]}.'
        assert parse_pairs(reply) == [("b", [(2, "y")])]

    def test_unusable_replies_give_nothing(self):
        assert parse_pairs("no json") == parse_pairs('{"citations": "x"}') == parse_pairs("[1]") == []


@pytest.mark.asyncio
class TestGround:
    async def test_quotes_become_offsets_and_cited_sources_are_numbered(self):
        registry = SourceRegistry()
        registry.add([from_web_result({"url": "https://other.nl"}), from_document(CHUNK)])
        agent = CitationAgent(
            citations(("Postal Address element is optional", [(2, "POSTAL ADDRESS 0..*, F"), (9, "x")]))
        )

        data = await ground(ANSWER, registry, agent)

        (citation,) = data.citations
        # Markdown inside the quote stays in the span: offsets are into the raw text.
        assert ANSWER[citation.start : citation.end] == "Postal Address** element is optional"
        (evidence,) = citation.evidence  # unknown number 9 dropped
        assert evidence.key == "doc:d1:c1"
        # The supporting passage, found in the source's own text.
        assert CHUNK.content[evidence.start : evidence.end] == evidence.text == "POSTAL ADDRESS 0..*, F"
        assert evidence.matcher == "exact"
        by_key = {s.key: s for s in data.sources}
        assert (by_key["doc:d1:c1"].number, by_key["doc:d1:c1"].cited) == (1, True)
        assert (by_key["url:https://other.nl"].number, by_key["url:https://other.nl"].cited) == (0, False)
        assert "[2] XAF_4.0.pdf · §4" in agent.calls[0]

    async def test_a_citation_whose_passage_is_not_in_the_source_is_rejected(self):
        registry = SourceRegistry()
        registry.add([from_document(CHUNK)])
        reply = citations(("It is not required", [(1, "something the source never says")]))

        data = await ground(ANSWER, registry, CitationAgent(reply))

        assert data.citations == []
        assert [s.key for s in data.sources] == ["doc:d1:c1"]  # still stored, uncited

    async def test_a_source_whose_passage_is_missing_is_dropped_when_another_is_found(self):
        registry = SourceRegistry()
        other = NumberedSource(key="doc:d2:c9", title="Other", content="unrelated")
        registry.add([from_document(CHUNK), other])
        reply = citations(("It is not required", [(1, "POSTAL ADDRESS 0..*"), (2, "POSTAL ADDRESS 0..*")]))

        (citation,) = (await ground(ANSWER, registry, CitationAgent(reply))).citations

        assert [e.key for e in citation.evidence] == ["doc:d1:c1"]

    async def test_a_paragraph_copied_with_a_sentence_left_out_is_found_by_its_sentences(self):
        policy = (
            "Strohm is committed to carrying out its business fairly. "
            "Accepting or paying bribes is a criminal offence. "
            "Strohm will suspend contracts with partners guilty of bribery. "
            "Contact the compliance officer with any concerns."
        )
        registry = SourceRegistry()
        registry.add([NumberedSource(key="doc:p:1", title="Policy", content=policy)])
        copied = (
            "Strohm is committed to carrying out its business fairly...\n\n"
            "Strohm will suspend contracts with partners guilty of bribery."
        )
        reply = citations(("It is not required", [(1, copied)]))

        (citation,) = (await ground(ANSWER, registry, CitationAgent(reply))).citations

        (evidence,) = citation.evidence
        assert evidence.matcher == "partial"
        assert evidence.text == policy[: policy.index(" Contact")]  # the source's own text

    async def test_an_earlier_turns_source_is_stored_only_when_cited(self):
        registry = SourceRegistry()
        registry.seed([
            NumberedSource(key="doc:d1:c1", title="XAF", content="POSTAL ADDRESS 0..*, F"),
            NumberedSource(key="doc:d2:c9", title="Other", content="unrelated"),
        ])
        reply = citations(("It is not required", [(1, "POSTAL ADDRESS")]))
        data = await ground(ANSWER, registry, CitationAgent(reply))
        assert [s.key for s in data.sources] == ["doc:d1:c1"]


@pytest.mark.django_db
@pytest.mark.asyncio
class TestStream:
    async def _thread(self) -> str:
        from django_ai_sdk.conversation.models import Thread

        return str((await Thread.objects.acreate()).id)

    def _stream(self, registry: SourceRegistry, thread_id: str, reply: str) -> Stream:
        stream = Stream.__new__(Stream)
        stream.citation_registry, stream.thread_id, stream.user = registry, thread_id, None
        stream.citation_agent = CitationAgent(reply)
        stream._persisted_tool_ids, stream._persisted_tool_output_ids = set(), set()
        return stream

    async def test_the_answer_is_followed_by_a_citations_artifact_and_seeds_the_next_turn(self):
        from django_ai_sdk.artifacts.models import Artifact

        thread_id = await self._thread()
        registry = SourceRegistry()
        registry.add([from_document(CHUNK)])
        stream = self._stream(registry, thread_id, citations(("element is optional", [(1, "POSTAL ADDRESS")])))
        writer = StreamWriter(message_id="m1")
        writer.add_chunk(stream.get_text_chunk(ANSWER))

        events = [e async for e in stream.get_citations_artifact(writer)]

        assert [e.event_type for e in events] == ["tool_call_start", "tool_input_complete", "tool_output"]
        payload = events[-1].tool_output
        (citation,) = payload["citations"]
        assert ANSWER[citation["start"] : citation["end"]] == "element is optional"
        assert payload["sources"][0]["documentId"] == "d1"
        assert await Artifact.objects.filter(thread_id=thread_id).acount() == 1
        assert writer.message.tool_calls[0]["name"] == "artifact_citations_artifact"

        # Next turn: no new search, the earlier source is still citable.
        later = SourceRegistry()
        await later.seed_from_thread(thread_id)
        assert [(s.key, s.content) for s in later.all] == [("doc:d1:c1", CHUNK.content)]

    async def test_no_sources_or_no_thread_means_no_artifact(self):
        thread_id = await self._thread()
        empty = self._stream(SourceRegistry(), thread_id, citations(("x", [(1, "x")])))
        registry = SourceRegistry()
        registry.add([from_document(CHUNK)])
        no_thread = self._stream(registry, "", citations(("x", [(1, "x")])))
        writer = StreamWriter(message_id="m1")

        assert [e async for e in empty.get_citations_artifact(writer)] == []
        assert [e async for e in no_thread.get_citations_artifact(writer)] == []

    async def test_a_failing_citation_agent_still_keeps_the_sources_for_later_turns(self):
        thread_id = await self._thread()
        registry = SourceRegistry()
        registry.add([from_document(CHUNK)])
        stream = self._stream(registry, thread_id, "")

        async def broken(*args, **kwargs):
            raise RuntimeError("provider down")

        stream.citation_agent.run = broken
        writer = StreamWriter(message_id="m1")
        writer.add_chunk(stream.get_text_chunk(ANSWER))

        events = [e async for e in stream.get_citations_artifact(writer)]

        assert events[-1].tool_output["citations"] == []
        later = SourceRegistry()
        await later.seed_from_thread(thread_id)
        assert [s.key for s in later.all] == ["doc:d1:c1"]


@pytest.mark.asyncio
async def test_the_stream_no_longer_rewrites_the_answer_text():
    # Bracket numbers or tags in the answer are just text now: nothing parses them.
    stream = Stream.__new__(Stream)
    stream.citation_registry = None
    queue: asyncio.Queue = asyncio.Queue()
    from haystack.dataclasses import StreamingChunk

    await queue.put(StreamingChunk(content='Optional [1] <source id="2" /> arr[2]'))
    await queue.put(_SENTINEL)
    events = [e async for e in stream.get_events(queue, None)]
    assert "".join(e.content for e in events) == 'Optional [1] <source id="2" /> arr[2]'


@pytest.fixture
def owner(django_user_model):
    return django_user_model.objects.create(email="owner@example.com")


@pytest.fixture
def stranger(django_user_model):
    return django_user_model.objects.create(email="stranger@example.com")


@pytest.mark.django_db
@pytest.mark.parametrize("url", ["/api/threads/{t}/sources/", "/drf/threads/{t}/sources/"])
def test_the_thread_lists_each_source_once(client, owner, stranger, url):
    from django_ai_sdk.artifacts.models import Artifact
    from django_ai_sdk.conversation.models import Thread

    thread = Thread.objects.create(user=owner)
    shared = {"key": "doc:d1:c1", "title": "XAF", "content": "POSTAL ADDRESS 0..*, F"}
    for cited in (False, True):  # two turns found the same chunk; the second cited it
        Artifact.objects.create(
            thread=thread,
            schema_name="CitationsArtifact",
            artifact_type="citations",
            data={"citations": [], "sources": [shared | {"cited": cited}]},
        )

    client.force_login(owner)
    response = client.get(url.format(t=thread.id))
    assert response.status_code == 200
    (source,) = response.json()["sources"]
    assert (source["key"], source["cited"]) == ("doc:d1:c1", True)

    client.force_login(stranger)
    assert client.get(url.format(t=thread.id)).status_code == 403
