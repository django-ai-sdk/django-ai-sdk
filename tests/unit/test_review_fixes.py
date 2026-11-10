"""Regression checks for the review/fable fixes."""

from __future__ import annotations

import json

import pytest
from django.core.files.base import ContentFile
from haystack.dataclasses import ChatMessage, Document, ToolCall

from django_ai_sdk.adapters.base import parse_tool_output
from django_ai_sdk.agents.subagent import _drop_handoff, _handoff_result_ids
from django_ai_sdk.contrib.ninja.schemas import FeedbackResponse
from django_ai_sdk.files.processors import get_mime_type, read_aio_bytes
from django_ai_sdk.rags.bm25 import BM25QueryExpanderRAG
from django_ai_sdk.rags.chroma import ChromaDBQueryExpanderRAG
from django_ai_sdk.rags.schemas import RagDocument


@pytest.mark.parametrize("text", ["{1, 2}", "b'x'", "...", "{(1, 2): 3}"])
def test_tool_output_stays_json_serializable(text):
    assert parse_tool_output(text) == text
    json.dumps(parse_tool_output({"a": text}))


def test_tool_output_still_parses_literals():
    assert parse_tool_output("{'a': 1}") == {"a": 1}


def test_subagent_result_drops_coordinator_context():
    call = ToolCall(tool_name="search", arguments={}, id="c1")
    handoff = [
        ChatMessage.from_user("hi"),
        ChatMessage.from_assistant(tool_calls=[call]),
        ChatMessage.from_tool("found", origin=call),
    ]
    own = ChatMessage.from_assistant("done")
    result = {"messages": [ChatMessage.from_system("sys"), *handoff, own]}

    assert _handoff_result_ids(handoff) == {"c1"}
    _drop_handoff(result, handoff)
    assert [m.text for m in result["messages"]] == ["sys", "done"]


def test_in_memory_chroma_instances_do_not_share_a_collection():
    a = ChromaDBQueryExpanderRAG([])._create_document_store()
    b = ChromaDBQueryExpanderRAG([])._create_document_store()
    a.write_documents([Document(content="x")])
    assert a.count_documents() == 1
    assert b.count_documents() == 0


async def test_bm25_documents_follow_add_and_remove():
    rag = BM25QueryExpanderRAG([RagDocument(id="1", content="one")])
    await rag.warmup()
    await rag.add_documents([RagDocument(id="1", content="uno"), RagDocument(id="2", content="two")])
    await rag.remove_documents(["2"])
    assert [(d.id, d.content) for d in rag.documents] == [("1", "uno")]


async def test_empty_and_pathless_files_are_readable():
    assert await get_mime_type(ContentFile(b"", name="empty.txt")) in (None, "text/plain")
    assert await read_aio_bytes(ContentFile(b"abc", name="a.txt")) == b"abc"


def test_feedback_without_id_validates():
    assert FeedbackResponse(rating=1, feedback="").id is None
