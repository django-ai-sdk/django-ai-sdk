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


# --- second round ------------------------------------------------------------


def test_system_prompt_braces_are_literal():
    from haystack.components.generators.chat import MockChatGenerator

    from django_ai_sdk.agents.tool_agent import ToolAgent

    prompt = 'Reply as {"x": {{1+1}}} to {{ customer }} {% if %}'
    agent = ToolAgent.build_agent(MockChatGenerator(), [], prompt)
    result = agent.run(messages=[ChatMessage.from_user("hi")])
    assert result["messages"][0].text == prompt


def test_tool_budget_refuses_a_batch_that_overshoots():
    from haystack.components.agents.state.state import State

    from django_ai_sdk.agents.tool_agent import ToolCallBudgetHook

    calls = [ToolCall(tool_name="t", arguments={}, id=f"c{i}") for i in range(3)]
    state = State(schema={"tool_call_counts": {"type": dict}})
    state.set("messages", [ChatMessage.from_assistant(tool_calls=calls)])
    state.set("tool_call_counts", {"t": 4})

    ToolCallBudgetHook(max_tool_calls=6).run(state)  # 2 left, 3 asked

    stubs = state.data["messages"][1:]
    assert len(stubs) == 3
    assert "at most 2" in stubs[0].tool_call_results[0].result


async def test_managing_a_code_agent_needs_staff():
    from types import SimpleNamespace

    from django_ai_sdk.permissions import AgentDefaultPermission, Operation

    agent = SimpleNamespace(is_runtime=False)
    user = SimpleNamespace(is_authenticated=True, is_staff=False, is_superuser=False)
    staff = SimpleNamespace(is_authenticated=True, is_staff=True, is_superuser=False)
    check = AgentDefaultPermission().has_permission

    assert not await check(user, Operation.UPDATE_AGENT, agent=agent)
    assert await check(staff, Operation.UPDATE_AGENT, agent=agent)
    assert await check(user, Operation.CREATE_AGENT)


def test_openai_content_shapes():
    from django_ai_sdk.protocols.openai import OpenAIProtocolHandler

    messages = OpenAIProtocolHandler().to_chat_messages(
        [
            {"role": "assistant", "content": None},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ]
    )
    assert [m.content for m in messages] == ["", "hi"]


@pytest.mark.django_db
def test_changing_an_mcp_server_url_forgets_its_grants(django_user_model):
    from django_ai_sdk.integrations.mcp.models import (
        MCPOAuthClient,
        MCPOAuthToken,
        MCPServerConfig,
    )

    user = django_user_model.objects.create(email="u@example.com")
    server = MCPServerConfig.objects.create(name="srv", url="https://a.example/mcp", auth="oauth")
    MCPOAuthToken.objects.create(user=user, server_name="srv", access_token="x")
    MCPOAuthClient.objects.create(server_name="srv", client_id="c")

    server.label = "renamed"
    server.save()
    assert MCPOAuthToken.objects.filter(server_name="srv").exists()

    server.url = "https://b.example/mcp"
    server.save()
    assert not MCPOAuthToken.objects.filter(server_name="srv").exists()
    assert not MCPOAuthClient.objects.filter(server_name="srv").exists()


@pytest.mark.django_db(transaction=True)
async def test_deleting_a_thread_removes_its_files(settings, tmp_path, mock_agents_registry):
    import os

    from asgiref.sync import sync_to_async

    from django_ai_sdk.artifacts import FileArtifact
    from django_ai_sdk.conversation.models import Thread
    from django_ai_sdk.memories.models import EntryDocument, Memory
    from django_ai_sdk.memories.services import MemoryService
    from django_ai_sdk.storage.db import DbStorageAdapter

    settings.MEDIA_ROOT = str(tmp_path)
    thread = await Thread.objects.acreate(metadata={"agent_id": "test-agent"})
    other = await Thread.objects.acreate(metadata={"agent_id": "test-agent"})
    memory = await MemoryService.get_or_create_thread_file_memory(str(thread.id))
    doc = EntryDocument(memory=memory, file_name="a.txt", content_type="text/plain")
    await sync_to_async(doc.file.save)("a.txt", ContentFile(b"x"), save=True)
    path = doc.file.path

    # The model cannot show a file of another thread, whatever url it makes up.
    files = {"files": [{"documentId": str(doc.id), "url": "https://evil.example/x"}]}
    with pytest.raises(ValueError, match="No file"):
        await FileArtifact.store(files, str(other.id))
    stored = await FileArtifact.store(files, str(thread.id))
    assert "evil.example" not in stored["files"][0]["url"]

    assert await DbStorageAdapter.delete_thread(str(thread.id))
    assert not await Memory.objects.filter(id=memory.id).aexists()
    assert not os.path.exists(path)
