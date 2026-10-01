"""Entry signals reach the cached RAG indexes of runtime agents, not only registry ones."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from django_ai_sdk.rags.provider import RAGProvider


@pytest.mark.django_db
@pytest.mark.asyncio
class TestEntrySignals:
    async def test_runtime_agent_cache_is_updated_once_per_class(self) -> None:
        from django_ai_sdk.agents.config import get_runtime_agent_class
        from django_ai_sdk.agents.models import AgentSettings
        from django_ai_sdk.memories.models import Entry, Memory

        memory = await Memory.objects.acreate(name="Docs")
        entry = await Entry.objects.acreate(memory=memory, content="hello", name="a.txt")
        # Two runtime agents of the same class share one cache entry.
        await AgentSettings.objects.acreate(name="A", model="gpt-4o-mini")
        await AgentSettings.objects.acreate(name="B", model="gpt-4o-mini")

        cls = get_runtime_agent_class(None)
        provider = RAGProvider()
        rag = MagicMock(
            spec=["add_documents", "remove_documents"],
            add_documents=AsyncMock(),
            remove_documents=AsyncMock(),
        )
        provider._cache[f"{cls.__name__}_{memory.id}"] = rag

        with patch.object(cls, "rag_provider", provider):
            entry.content = "updated"
            await entry.asave()
            rag.add_documents.assert_awaited_once()
            assert rag.add_documents.await_args.args[0][0].id == str(entry.id)

            entry_id = str(entry.id)
            await entry.adelete()
            rag.remove_documents.assert_awaited_once_with([entry_id])
