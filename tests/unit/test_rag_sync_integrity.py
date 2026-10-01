"""The index must end up holding exactly the source documents, whatever happened to them.

Real Qdrant (on disk), no mocks of the index: each test changes the documents the way
production does (another process, the signal handler in this process, a crash halfway)
and then asserts on what the index actually contains.
"""

from __future__ import annotations

import pytest

from django_ai_sdk.rags.chroma import ChromaDBQueryExpanderRAG, ChromaDBQueryExpanderRAGConfig
from django_ai_sdk.rags.config import ChromaStorageConfig, QdrantStorageConfig
from django_ai_sdk.rags.qdrant_hybrid import QdrantBM25HybridRAG, QdrantBM25HybridRAGConfig
from django_ai_sdk.rags.schemas import RagDocument


BACKEND = {"name": "qdrant"}


@pytest.fixture(params=["qdrant", "chroma"], autouse=True)
def backend(request):
    BACKEND["name"] = request.param
    return request.param


def make_rag(path, documents):
    if BACKEND["name"] == "chroma":
        storage = ChromaStorageConfig(backend="persistent", persist_path=str(path))
        return ChromaDBQueryExpanderRAG(
            documents=documents, config=ChromaDBQueryExpanderRAGConfig(storage=storage)
        )
    storage = QdrantStorageConfig(backend="persistent", persist_path=str(path))
    return QdrantBM25HybridRAG(
        documents=documents, config=QdrantBM25HybridRAGConfig(storage=storage)
    )


def chunks(rag):
    """Everything in the index as (doc_id, content, name) tuples."""
    return sorted(
        (c.meta.get("doc_id"), c.content, c.meta.get("name"))
        for c in rag._cached_document_store.filter_documents()
    )


def doc(doc_id, content, name=None):
    return RagDocument(id=doc_id, content=content, metadata={"name": name} if name else {})


A = doc("a", "Invoice from Acme for hosting", "acme.pdf")
B = doc("b", "Invoice from Globex for design work", "globex.pdf")


@pytest.fixture
async def warm(tmp_path):
    rag = make_rag(tmp_path, [A])
    await rag.warmup()
    return rag


@pytest.mark.asyncio
async def test_emptied_then_refilled_index_gets_the_new_documents(warm):
    await warm.sync_documents([])
    assert chunks(warm) == []

    await warm.sync_documents([B])

    assert [c[0] for c in chunks(warm)] == ["b"]


@pytest.mark.asyncio
async def test_an_edit_made_in_this_process_leaves_no_old_chunks_behind(warm):
    """The signal handler calls add_documents for an edited entry, nothing removes the old text."""
    edited = doc("a", "Credit note from Acme", "acme.pdf")

    await warm.add_documents([edited])
    await warm.sync_documents([edited])

    assert [c[1] for c in chunks(warm)] == ["Credit note from Acme"]


@pytest.mark.asyncio
async def test_renaming_a_document_updates_its_metadata(warm):
    """Entry metadata (name, keywords, facts) is shown in citations and embedded for retrieval."""
    renamed = doc("a", "Invoice from Acme for hosting", "acme-2026.pdf")

    await warm.sync_documents([renamed])

    assert {c[2] for c in chunks(warm)} == {"acme-2026.pdf"}


@pytest.mark.asyncio
async def test_syncing_twice_changes_nothing(warm):
    await warm.sync_documents([A, B])
    first = chunks(warm)

    await warm.sync_documents([A, B])

    assert chunks(warm) == first


@pytest.mark.asyncio
async def test_a_failed_sync_never_loses_the_documents_it_was_replacing(warm, monkeypatch):
    """If embedding the new text fails, the old text must still be searchable."""
    edited = doc("a", "Credit note from Acme", "acme.pdf")

    async def boom(*args, **kwargs):
        raise RuntimeError("embedder down")

    monkeypatch.setattr(warm, "_index_documents", boom)
    await warm.sync_or_keep([edited])

    assert warm.stale is True
    assert [c[1] for c in chunks(warm)] == ["Invoice from Acme for hosting"]


@pytest.mark.asyncio
async def test_a_stale_index_catches_up_on_the_next_sync(warm, monkeypatch):
    edited = doc("a", "Credit note from Acme", "acme.pdf")
    real = warm._index_documents

    async def boom(*args, **kwargs):
        raise RuntimeError("embedder down")

    monkeypatch.setattr(warm, "_index_documents", boom)
    await warm.sync_or_keep([edited, B])
    monkeypatch.setattr(warm, "_index_documents", real)
    await warm.sync_or_keep([edited, B])

    assert warm.stale is False
    assert [c[:2] for c in chunks(warm)] == [
        ("a", "Credit note from Acme"),
        ("b", "Invoice from Globex for design work"),
    ]


@pytest.mark.asyncio
async def test_a_chunked_document_is_replaced_as_a_whole(tmp_path):
    long_old = doc("a", " ".join(f"old sentence {i}." for i in range(400)))
    long_new = doc("a", " ".join(f"new sentence {i}." for i in range(400)))
    rag = make_rag(tmp_path, [long_old])
    await rag.warmup()
    assert len(chunks(rag)) > 1

    await rag.sync_documents([long_new])

    assert all("new sentence" in c[1] for c in chunks(rag))


@pytest.mark.asyncio
async def test_an_index_from_before_doc_version_is_upgraded_without_duplicates(tmp_path):
    """Chunks written by the previous release carry doc_id but no doc_version."""
    rag = make_rag(tmp_path, [A])
    await rag.warmup()
    store = rag._cached_document_store
    legacy = store.filter_documents()
    for chunk in legacy:
        chunk.meta.pop("doc_version", None)
    store.write_documents(
        legacy, policy=__import__("haystack").document_stores.types.DuplicatePolicy.OVERWRITE
    )

    await rag.sync_documents([A])

    assert [c[:2] for c in chunks(rag)] == [("a", "Invoice from Acme for hosting")]
