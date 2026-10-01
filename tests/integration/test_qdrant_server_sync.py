"""Several web processes syncing one Qdrant server collection at the same time.

Needs a Qdrant server: set QDRANT_TEST_URL (e.g. http://localhost:6333). Uses its own
throwaway collection and deletes it afterwards.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import httpx
import pytest

from django_ai_sdk.rags.config import QdrantStorageConfig
from django_ai_sdk.rags.qdrant_hybrid import QdrantBM25HybridRAG, QdrantBM25HybridRAGConfig
from django_ai_sdk.rags.schemas import RagDocument

URL = os.environ.get("QDRANT_TEST_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set QDRANT_TEST_URL to run against a server")


def doc(doc_id: str, content: str) -> RagDocument:
    return RagDocument(id=doc_id, content=content, metadata={"name": f"{doc_id}.txt"})


@pytest.fixture
def collection():
    name = f"rag_sync_test_{uuid.uuid4().hex}"
    yield name
    httpx.delete(f"{URL}/collections/{name}", timeout=10)


def make_rag(collection: str, documents: list[RagDocument]) -> QdrantBM25HybridRAG:
    storage = QdrantStorageConfig(backend="server", location=URL, index=collection)
    return QdrantBM25HybridRAG(
        documents=documents, config=QdrantBM25HybridRAGConfig(storage=storage)
    )


def held(rag: QdrantBM25HybridRAG) -> list[tuple[str, str]]:
    return sorted(
        (c.meta["doc_id"], c.content) for c in rag._cached_document_store.filter_documents()
    )


@pytest.mark.asyncio
async def test_two_processes_syncing_the_same_changes_leave_no_duplicates(collection):
    start = [doc("a", "Invoice from Acme"), doc("b", "Invoice from Globex")]
    web_1, web_2 = make_rag(collection, start), make_rag(collection, start)
    await web_1.warmup()
    await web_2.warmup()

    target = [doc("a", "Credit note from Acme"), doc("c", "Invoice from Initech")]
    await asyncio.gather(web_1.sync_documents(target), web_2.sync_documents(target))

    expected = [("a", "Credit note from Acme"), ("c", "Invoice from Initech")]
    assert held(web_1) == held(web_2) == expected


@pytest.mark.asyncio
async def test_processes_that_disagree_converge_once_they_agree(collection):
    start = [doc("a", "Invoice from Acme")]
    web_1, web_2 = make_rag(collection, start), make_rag(collection, start)
    await web_1.warmup()
    await web_2.warmup()

    older = [doc("a", "Invoice from Acme, draft")]
    newer = [doc("a", "Invoice from Acme, final")]
    await asyncio.gather(web_1.sync_documents(older), web_2.sync_documents(newer))
    await web_1.sync_documents(newer)
    await web_2.sync_documents(newer)

    assert held(web_1) == [("a", "Invoice from Acme, final")]


@pytest.mark.asyncio
async def test_a_collection_emptied_and_refilled_through_the_server(collection):
    web = make_rag(collection, [doc("a", "Invoice from Acme")])
    await web.warmup()

    await web.sync_documents([])
    assert held(web) == []
    await web.sync_documents([doc("z", "Brand new upload")])

    assert held(web) == [("z", "Brand new upload")]
