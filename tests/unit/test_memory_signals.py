"""Entry signals reach the cached RAG indexes of runtime agents, not only registry ones."""

from __future__ import annotations

from contextlib import nullcontext
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
        rag = MagicMock(add_documents=AsyncMock(), remove_documents=AsyncMock())
        provider._cache[f"{cls.__name__}_{memory.id}"] = rag

        with patch.object(cls, "rag_provider", provider):
            entry.content = "updated"
            await entry.asave()
            rag.add_documents.assert_awaited_once()
            assert rag.add_documents.await_args.args[0][0].id == str(entry.id)

            entry_id = str(entry.id)
            await entry.adelete()
            rag.remove_documents.assert_awaited_once_with([entry_id])


SIGNATURE = {"sdk": {"rag": {"dim": 384}}, "owner": "hr"}


@pytest.mark.django_db
@pytest.mark.asyncio
class TestIndexSignature:
    """A change the index missed drops the stored signature, so reindex rebuilds it."""

    async def _save_entry(self, add_documents: AsyncMock) -> dict:
        from django_ai_sdk.memories.models import Entry, Memory

        memory = await Memory.objects.acreate(name="Docs", metadata=SIGNATURE)
        agent = MagicMock(rag_provider=MagicMock(add_documents=add_documents))
        with (
            patch(
                "django_ai_sdk.agents.services.AgentService.get_rag_agents",
                AsyncMock(return_value=[agent]),
            ),
            pytest.raises(RuntimeError) if add_documents.side_effect else nullcontext(),
        ):
            await Entry.objects.acreate(memory=memory, content="hello", name="a.txt")
        await memory.arefresh_from_db()
        return memory.metadata

    async def test_an_indexed_entry_keeps_the_signature(self) -> None:
        assert await self._save_entry(AsyncMock(return_value=True)) == SIGNATURE

    async def test_a_failed_index_write_drops_it(self) -> None:
        metadata = await self._save_entry(AsyncMock(side_effect=RuntimeError("qdrant down")))
        assert metadata == {"sdk": {}, "owner": "hr"}

    async def test_an_unwritten_index_drops_it(self) -> None:
        metadata = await self._save_entry(AsyncMock(return_value=False))
        assert metadata == {"sdk": {}, "owner": "hr"}
