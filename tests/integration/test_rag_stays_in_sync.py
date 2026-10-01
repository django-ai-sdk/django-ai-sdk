"""End to end: the RAG a chat uses always matches the memory's documents.

Real database rows, a real on-disk Qdrant index and the real RAGProvider. The web
process holds a warm index; the worker (another process in production) changes the
documents behind its back, and the next use of the index has to notice.
"""

from __future__ import annotations

import pytest

from django_ai_sdk.memories.models import Entry, Memory
from django_ai_sdk.rags.config import QdrantStorageConfig
from django_ai_sdk.rags.provider import RAGProvider
from django_ai_sdk.rags.qdrant_hybrid import QdrantBM25HybridRAG, QdrantBM25HybridRAGConfig
from django_ai_sdk.rags.utils import queryset_to_rag_documents


class SearchAgent:
    """What the SDK's runtime agent does for RAG, minus the LLM."""

    def __init__(self, path) -> None:
        self.path = path

    async def get_rag_queryset(self, memory_id=None):
        fields = ("id", "content", "data", "name", "memory_id")
        return Entry.objects.filter(memory_id=memory_id).only(*fields).order_by("-updated_at")

    async def get_rag_pipeline(self, memory_id=None):
        documents = await queryset_to_rag_documents(await self.get_rag_queryset(memory_id))
        if not documents:
            return None
        storage = QdrantStorageConfig(backend="persistent", persist_path=str(self.path))
        return QdrantBM25HybridRAG(
            documents=documents, config=QdrantBM25HybridRAGConfig(storage=storage)
        )


def in_index(rag) -> list[tuple[str, str, str]]:
    return sorted(
        (c.meta["doc_id"], c.meta["name"], c.content)
        for c in rag._cached_document_store.filter_documents()
    )


async def in_database(memory) -> list[tuple[str, str, str]]:
    return sorted([(str(e.id), e.name, e.content) async for e in memory.entries.all()])


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
class TestTheIndexFollowsTheDocuments:
    async def setup_memory(self, tmp_path):
        memory = await Memory.objects.acreate(name="Docs")
        for name, content in [("a.txt", "Invoice from Acme"), ("b.txt", "Invoice from Globex")]:
            await Entry.objects.acreate(memory=memory, name=name, content=content)
        agent, web = SearchAgent(tmp_path), RAGProvider()
        rag = await web.get_rag_instance(agent, str(memory.id))
        assert in_index(rag) == await in_database(memory)
        return memory, agent, web, rag

    async def assert_in_sync(self, memory, agent, web, rag) -> None:
        current = await web.get_rag_instance(agent, str(memory.id))
        assert current is rag, "the warm index should be kept and synced, not replaced"
        assert in_index(current) == await in_database(memory)

    async def test_a_document_added_by_the_worker_is_searchable(self, tmp_path):
        memory, agent, web, rag = await self.setup_memory(tmp_path)
        await Entry.objects.acreate(memory=memory, name="c.txt", content="Invoice from Initech")
        await self.assert_in_sync(memory, agent, web, rag)

    async def test_a_document_edited_by_the_worker_is_replaced(self, tmp_path):
        memory, agent, web, rag = await self.setup_memory(tmp_path)
        entry = await memory.entries.aget(name="a.txt")
        entry.content = "Credit note from Acme"
        await entry.asave()
        await self.assert_in_sync(memory, agent, web, rag)

    async def test_a_renamed_document_shows_its_new_name(self, tmp_path):
        memory, agent, web, rag = await self.setup_memory(tmp_path)
        entry = await memory.entries.aget(name="a.txt")
        entry.name = "acme-2026.txt"
        await entry.asave()
        await self.assert_in_sync(memory, agent, web, rag)

    async def test_a_document_deleted_by_the_worker_is_gone(self, tmp_path):
        memory, agent, web, rag = await self.setup_memory(tmp_path)
        await (await memory.entries.aget(name="b.txt")).adelete()
        await self.assert_in_sync(memory, agent, web, rag)

    async def test_everything_deleted_and_then_a_new_upload(self, tmp_path):
        memory, agent, web, rag = await self.setup_memory(tmp_path)
        await memory.entries.all().adelete()
        # nothing left: the web process keeps asking, and must not serve the deleted text
        current = await web.get_rag_instance(agent, str(memory.id))
        assert current is None or in_index(current) == []
        await Entry.objects.acreate(memory=memory, name="new.txt", content="Brand new upload")

        current = await web.get_rag_instance(agent, str(memory.id))

        assert [row[1] for row in in_index(current)] == ["new.txt"]

    async def test_many_changes_in_a_row_end_exactly_matching(self, tmp_path):
        memory, agent, web, rag = await self.setup_memory(tmp_path)
        for step in range(5):
            await Entry.objects.acreate(memory=memory, name=f"n{step}.txt", content=f"note {step}")
            entry = await memory.entries.aget(name="a.txt")
            entry.content = f"Invoice from Acme, revision {step}"
            await entry.asave()
            if step % 2:
                await (await memory.entries.aget(name=f"n{step - 1}.txt")).adelete()
            await self.assert_in_sync(memory, agent, web, rag)

    async def test_an_edit_saved_in_the_web_process_itself(self, tmp_path):
        """The signal handler path: the cached index is updated in place, then must still agree."""
        memory, agent, web, rag = await self.setup_memory(tmp_path)
        entry = await memory.entries.aget(name="a.txt")
        entry.content = "Credit note from Acme"
        await entry.asave()
        await web.add_documents(
            agent,
            str(memory.id),
            await queryset_to_rag_documents(Entry.objects.filter(id=entry.id)),
        )

        await self.assert_in_sync(memory, agent, web, rag)

    async def test_two_web_processes_syncing_the_same_changes(self, tmp_path):
        """Two requests hit a stale index at once: one syncs, none double-adds."""
        import asyncio

        memory, agent, web, rag = await self.setup_memory(tmp_path)
        await Entry.objects.acreate(memory=memory, name="c.txt", content="Invoice from Initech")
        entry = await memory.entries.aget(name="a.txt")
        entry.content = "Credit note from Acme"
        await entry.asave()

        await asyncio.gather(*[web.get_rag_instance(agent, str(memory.id)) for _ in range(5)])

        await self.assert_in_sync(memory, agent, web, rag)
