from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from django_ai_sdk import Assistant
from django_ai_sdk.memories.models import Entry, ThreadMemory
from django_ai_sdk.protocols.vercel import VercelProtocolHandler
from django_ai_sdk.storage.memory import MemoryStorageAdapter
from tests.factories.db import MemoryFactory


class RagToolsAssistant(Assistant):
    name = "rag_tools_test"
    model = "gpt-4o-mini"
    instructions = ["You are a test assistant"]
    protocol = VercelProtocolHandler
    storage_adapter = MemoryStorageAdapter

    async def get_pipeline_adapter(self, thread_id=None, user=None):
        pass


@pytest.mark.django_db
@pytest.mark.asyncio
class TestAssistantGetRagTools:
    """Tests for Assistant.get_rag_tools()."""

    async def test_returns_empty_when_no_rag_provider(self):
        assistant = RagToolsAssistant()
        assert assistant.rag_provider is None
        result = await assistant.get_rag_tools(thread_id="any-thread")
        assert result == []

    async def test_returns_empty_when_no_thread_id(self):
        assistant = RagToolsAssistant()
        assistant.rag_provider = MagicMock()
        result = await assistant.get_rag_tools(thread_id="")
        assert result == []

    async def test_empty_thread_id_allows_none_check_with_provider(self):
        assistant = RagToolsAssistant()
        mock_provider = MagicMock()
        mock_provider.get_tool = AsyncMock()
        assistant.rag_provider = mock_provider
        result = await assistant.get_rag_tools(thread_id="")
        assert result == []
        mock_provider.get_tool.assert_not_called()

    async def test_returns_tool_for_active_memory_with_docs(self):
        memory = await MemoryFactory.acreate(name="HR Knowledge")
        await Entry.objects.acreate(memory=memory, content="doc 1")
        await Entry.objects.acreate(memory=memory, content="doc 2")

        from django_ai_sdk.conversation.models import Thread

        thread = await Thread.objects.acreate()
        await ThreadMemory.objects.acreate(thread=thread, memory=memory, active=True)

        assistant = RagToolsAssistant()
        mock_tool = MagicMock()
        mock_provider = MagicMock()
        mock_provider.get_tool = AsyncMock(return_value=mock_tool)
        assistant.rag_provider = mock_provider

        result = await assistant.get_rag_tools(thread_id=str(thread.id))

        assert len(result) == 1
        assert result[0] is mock_tool

    async def test_skips_inactive_memory(self):
        memory = await MemoryFactory.acreate(name="Inactive Mem")
        await Entry.objects.acreate(memory=memory, content="doc")

        from django_ai_sdk.conversation.models import Thread

        thread = await Thread.objects.acreate()
        await ThreadMemory.objects.acreate(thread=thread, memory=memory, active=False)

        assistant = RagToolsAssistant()
        mock_provider = MagicMock()
        mock_provider.get_tool = AsyncMock()
        assistant.rag_provider = mock_provider

        result = await assistant.get_rag_tools(thread_id=str(thread.id))

        assert result == []
        mock_provider.get_tool.assert_not_called()

    async def test_logs_warning_for_empty_memory(self):
        memory = await MemoryFactory.acreate(name="Empty Knowledge")
        from django_ai_sdk.conversation.models import Thread

        thread = await Thread.objects.acreate()
        await ThreadMemory.objects.acreate(thread=thread, memory=memory, active=True)

        assistant = RagToolsAssistant()
        mock_provider = MagicMock()
        mock_provider.get_tool = AsyncMock(return_value=None)
        assistant.rag_provider = mock_provider

        result = await assistant.get_rag_tools(thread_id=str(thread.id))

        assert result == []
        mock_provider.get_tool.assert_awaited_once()

    async def test_multiple_active_memories_all_returned(self):
        mem_a = await MemoryFactory.acreate(name="Base de Connaissances")
        mem_b = await MemoryFactory.acreate(name="SAV Procedure")
        await Entry.objects.acreate(memory=mem_a, content="faq")
        await Entry.objects.acreate(memory=mem_a, content="guide")
        await Entry.objects.acreate(memory=mem_b, content="refund")
        await Entry.objects.acreate(memory=mem_b, content="exchange")

        from django_ai_sdk.conversation.models import Thread

        thread = await Thread.objects.acreate()
        await ThreadMemory.objects.acreate(thread=thread, memory=mem_a, active=True)
        await ThreadMemory.objects.acreate(thread=thread, memory=mem_b, active=True)

        assistant = RagToolsAssistant()
        tool_a = MagicMock()
        tool_b = MagicMock()
        mock_provider = MagicMock()
        mock_provider.get_tool = AsyncMock(side_effect=[tool_a, tool_b])
        assistant.rag_provider = mock_provider

        result = await assistant.get_rag_tools(thread_id=str(thread.id))

        assert len(result) == 2
        assert result[0] is tool_a
        assert result[1] is tool_b

    async def test_mixed_active_inactive_and_empty(self):
        active = await MemoryFactory.acreate(name="Active")
        inactive = await MemoryFactory.acreate(name="Inactive")
        empty = await MemoryFactory.acreate(name="Empty")
        await Entry.objects.acreate(memory=active, content="doc")
        await Entry.objects.acreate(memory=inactive, content="doc")

        from django_ai_sdk.conversation.models import Thread

        thread = await Thread.objects.acreate()
        await ThreadMemory.objects.acreate(thread=thread, memory=active, active=True)
        await ThreadMemory.objects.acreate(thread=thread, memory=inactive, active=False)
        await ThreadMemory.objects.acreate(thread=thread, memory=empty, active=True)

        assistant = RagToolsAssistant()
        mock_tool = MagicMock()
        mock_provider = MagicMock()

        async def get_tool_side_effect(_assistant, memory_id, **kwargs):
            return None if memory_id == str(empty.id) else mock_tool

        mock_provider.get_tool = AsyncMock(side_effect=get_tool_side_effect)
        assistant.rag_provider = mock_provider

        result = await assistant.get_rag_tools(thread_id=str(thread.id))

        assert len(result) == 1
        assert result[0] is mock_tool

    async def test_passes_citation_registry_to_provider(self):
        memory = await MemoryFactory.acreate(name="Cited Memory")
        await Entry.objects.acreate(memory=memory, content="cited doc")

        from django_ai_sdk.conversation.models import Thread

        thread = await Thread.objects.acreate()
        await ThreadMemory.objects.acreate(thread=thread, memory=memory, active=True)

        assistant = RagToolsAssistant()
        mock_tool = MagicMock()
        mock_provider = MagicMock()
        mock_provider.get_tool = AsyncMock(return_value=mock_tool)
        assistant.rag_provider = mock_provider

        citation_registry = MagicMock()
        citation_formatter = MagicMock()

        result = await assistant.get_rag_tools(
            thread_id=str(thread.id),
            citation_registry=citation_registry,
            citation_formatter=citation_formatter,
        )

        assert len(result) == 1
        call_kwargs = mock_provider.get_tool.await_args.kwargs
        assert call_kwargs["citation_registry"] is citation_registry
        assert call_kwargs["citation_formatter"] is citation_formatter

    async def test_passes_correct_memory_id_and_spec(self):
        memory = await MemoryFactory.acreate(name="Special Knowledge")
        await Entry.objects.acreate(memory=memory, content="unique doc")

        from django_ai_sdk.conversation.models import Thread

        thread = await Thread.objects.acreate()
        await ThreadMemory.objects.acreate(thread=thread, memory=memory, active=True)

        assistant = RagToolsAssistant()
        mock_tool = MagicMock()
        mock_provider = MagicMock()
        mock_provider.get_tool = AsyncMock(return_value=mock_tool)
        assistant.rag_provider = mock_provider

        await assistant.get_rag_tools(thread_id=str(thread.id))

        call_args = mock_provider.get_tool.await_args
        assert call_args.args[1] == str(memory.id)
        spec = call_args.kwargs["spec"]
        assert spec.name == "search_special_knowledge"
        assert spec.doc_count == 1
