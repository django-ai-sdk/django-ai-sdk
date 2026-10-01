"""A warm index follows documents that another process added, edited or deleted.

An upload is processed by the worker, which has no cached index to update. The web
process keeps its own warm index, so a cache hit has to notice the change itself.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from django_ai_sdk.rags.base import RAGBase
from django_ai_sdk.rags.bm25 import BM25QueryExpanderRAG
from django_ai_sdk.rags.provider import RAGProvider
from django_ai_sdk.rags.schemas import RagDocument

DOCS = [RagDocument(id="acme", content="Invoice from Acme")]
MORE_DOCS = [*DOCS, RagDocument(id="globex", content="Invoice from Globex")]


def warm_rag() -> MagicMock:
    rag = MagicMock(spec=["needs_warmup", "warmup", "sync_documents"])
    rag.needs_warmup = False
    rag.sync_documents = AsyncMock()
    return rag


def agent_for(rag: object, *later_documents: list[RagDocument]) -> MagicMock:
    """Builds `rag` first, then a throwaway pipeline whose documents are the current ones."""
    agent = MagicMock()
    agent.__class__.__name__ = "MockAgent"
    fresh = [SimpleNamespace(documents=docs) for docs in later_documents]
    agent.get_rag_pipeline = AsyncMock(side_effect=[rag, *fresh])
    return agent


@pytest.mark.asyncio
async def test_an_unchanged_memory_is_not_synced() -> None:
    rag, provider = warm_rag(), RAGProvider()
    agent = agent_for(rag)

    with patch.object(RAGProvider, "_fingerprint", AsyncMock(return_value=(1, "t1"))):
        await provider.get_rag_instance(agent, "m1")
        await provider.get_rag_instance(agent, "m1")

    rag.sync_documents.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_change_made_elsewhere_syncs_the_warm_index_once() -> None:
    rag, provider = warm_rag(), RAGProvider()
    agent = agent_for(rag, MORE_DOCS)
    fingerprints = AsyncMock(side_effect=[(1, "t1"), (2, "t2"), (2, "t2")])

    with patch.object(RAGProvider, "_fingerprint", fingerprints):
        await provider.get_rag_instance(agent, "m1")  # builds the index
        await provider.get_rag_instance(agent, "m1")  # a document appeared meanwhile
        await provider.get_rag_instance(agent, "m1")  # nothing new

    rag.sync_documents.assert_awaited_once_with(MORE_DOCS)


@pytest.mark.asyncio
async def test_a_failed_sync_keeps_the_index_and_is_tried_again() -> None:
    rag, provider = warm_rag(), RAGProvider()
    rag.sync_documents.side_effect = [RuntimeError("store locked"), None]
    agent = agent_for(rag, MORE_DOCS, MORE_DOCS)
    fingerprints = AsyncMock(side_effect=[(1, "t1"), (2, "t2"), (2, "t2")])

    with patch.object(RAGProvider, "_fingerprint", fingerprints):
        await provider.get_rag_instance(agent, "m1")
        assert await provider.get_rag_instance(agent, "m1") is rag
        await provider.get_rag_instance(agent, "m1")

    assert rag.sync_documents.await_count == 2


@pytest.mark.asyncio
async def test_an_index_opened_stale_is_synced_on_its_next_use() -> None:
    rag, provider = warm_rag(), RAGProvider()
    rag.stale = True
    agent = agent_for(rag, MORE_DOCS)

    with patch.object(RAGProvider, "_fingerprint", AsyncMock(return_value=(1, "t1"))):
        await provider.get_rag_instance(agent, "m1")
        await provider.get_rag_instance(agent, "m1")

    rag.sync_documents.assert_awaited_once_with(MORE_DOCS)


@pytest.mark.asyncio
async def test_sync_or_keep_marks_the_index_stale_instead_of_raising() -> None:
    index = SimpleNamespace(sync_documents=AsyncMock(side_effect=RuntimeError("locked")))

    await RAGBase.sync_or_keep(index, DOCS)
    assert index.stale is True

    index.sync_documents.side_effect = None
    await RAGBase.sync_or_keep(index, DOCS)
    assert index.stale is False


@pytest.mark.asyncio
async def test_an_unknown_fingerprint_never_triggers_a_sync() -> None:
    rag, provider = warm_rag(), RAGProvider()
    agent = agent_for(rag)

    with patch.object(RAGProvider, "_fingerprint", AsyncMock(return_value=None)):
        await provider.get_rag_instance(agent, "m1")
        await provider.get_rag_instance(agent, "m1")

    rag.sync_documents.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_rag_that_cannot_sync_is_left_alone() -> None:
    rag = MagicMock(spec=["needs_warmup", "warmup"])
    rag.needs_warmup = False
    provider, agent = RAGProvider(), agent_for(rag)
    fingerprints = AsyncMock(side_effect=[(1, "t1"), (2, "t2")])

    with patch.object(RAGProvider, "_fingerprint", fingerprints):
        await provider.get_rag_instance(agent, "m1")
        assert await provider.get_rag_instance(agent, "m1") is rag


@pytest.mark.asyncio
async def test_changed_documents_are_read_the_way_the_agent_builds_them() -> None:
    """A custom get_rag_pipeline supplies its own documents; the sync must use them."""
    rag, provider = warm_rag(), RAGProvider()
    custom = [RagDocument(id="custom", content="from the agent's own source")]
    agent = agent_for(rag, custom)
    fingerprints = AsyncMock(side_effect=[(1, "t1"), (2, "t2")])

    with patch.object(RAGProvider, "_fingerprint", fingerprints):
        await provider.get_rag_instance(agent, "m1")
        await provider.get_rag_instance(agent, "m1")

    rag.sync_documents.assert_awaited_once_with(custom)


@pytest.mark.asyncio
async def test_an_in_memory_rag_syncs_too() -> None:
    """The sync lives on RAGBase, so it is not Qdrant-only."""
    rag = BM25QueryExpanderRAG(documents=DOCS)
    await rag.warmup()

    await rag.sync_documents(MORE_DOCS)

    assert rag.documents == MORE_DOCS
    assert rag._cached_document_store.count_documents() == 2


@pytest.mark.django_db
@pytest.mark.asyncio
class TestFingerprint:
    """The real check, against the entries table."""

    @staticmethod
    def agent() -> object:
        from django_ai_sdk import Agent
        from django_ai_sdk.protocols.vercel import VercelProtocolHandler
        from django_ai_sdk.storage.memory import MemoryStorageAdapter

        class FingerprintAgent(Agent):
            name = "fingerprint_test"
            model = "gpt-4o-mini"
            instructions = ["test"]
            protocol = VercelProtocolHandler
            storage_adapter = MemoryStorageAdapter

            async def get_pipeline_adapter(self, thread_id=None, user=None):
                pass

        return FingerprintAgent()

    async def test_it_changes_when_a_document_is_added_edited_or_deleted(self) -> None:
        from django_ai_sdk.memories.models import Entry, Memory

        provider, agent = RAGProvider(), self.agent()
        memory = await Memory.objects.acreate(name="Docs")
        other = await Memory.objects.acreate(name="Other")
        seen = [await provider._fingerprint(agent, str(memory.id))]

        entry = await Entry.objects.acreate(memory=memory, content="a", name="a.txt")
        seen.append(await provider._fingerprint(agent, str(memory.id)))

        entry.content = "edited"
        await entry.asave()
        seen.append(await provider._fingerprint(agent, str(memory.id)))

        await Entry.objects.acreate(memory=other, content="elsewhere", name="b.txt")
        assert await provider._fingerprint(agent, str(memory.id)) == seen[-1]  # other memory

        await entry.adelete()
        seen.append(await provider._fingerprint(agent, str(memory.id)))

        # Every step differs from the one before it (a delete lands back on an empty memory).
        assert all(before != after for before, after in zip(seen, seen[1:]))


@pytest.mark.asyncio
async def test_a_memory_without_documents_is_not_remembered_as_empty() -> None:
    """The first upload to a memory that was opened empty must still get indexed."""
    rag, provider = warm_rag(), RAGProvider()
    agent = MagicMock()
    agent.__class__.__name__ = "MockAgent"
    agent.get_rag_pipeline = AsyncMock(side_effect=[None, rag])

    with patch.object(RAGProvider, "_fingerprint", AsyncMock(return_value=(0, None))):
        assert await provider.get_rag_instance(agent, "m1") is None  # no documents yet
        assert await provider.get_rag_instance(agent, "m1") is rag  # after the first upload
