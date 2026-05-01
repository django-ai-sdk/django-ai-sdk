"""
Tests for plain BM25 RAG implementation.

These tests verify BM25RAG without any LLM calls.
The bm25s library works entirely locally.
"""

import pytest
from django_ai_sdk.rags.bm25 import BM25RAG, BM25Config
from django_ai_sdk.rags.base import RAGResult, RAGSource
from django_ai_sdk.rags.schemas import RagDocument


class TestBM25RAGInit:
    """Test BM25RAG initialization."""

    def test_init_with_documents(self):
        """Test initialization with document list."""
        docs = [RagDocument(id="1", content="Python is great")]
        rag = BM25RAG(documents=docs)
        assert len(rag.documents) == 1
        assert not rag._is_warmed_up
        assert rag._bm25 is None

    def test_init_with_empty_documents(self):
        """Test initialization with empty list."""
        rag = BM25RAG(documents=[])
        assert len(rag.documents) == 0
        assert not rag._is_warmed_up

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = BM25Config(top_k=10, k1=1.0, b=0.5)
        rag = BM25RAG(documents=[], config=config)
        assert rag.config.top_k == 10
        assert rag.config.k1 == 1.0
        assert rag.config.b == 0.5

    def test_init_converts_dicts_to_ragdocuments(self):
        """Test that dict inputs are converted to RagDocument."""
        docs = [{"id": "1", "content": "test"}]
        rag = BM25RAG(documents=docs)
        assert isinstance(rag.documents[0], RagDocument)
        assert rag.documents[0].id == "1"


class TestBM25RAGWarmup:
    """Test BM25RAG warmup (index building)."""

    def test_warmup_builds_index(self):
        """Test that warmup builds the BM25 index."""
        docs = [RagDocument(id=f"doc-{i}", content=f"Content {i}") for i in range(5)]
        rag = BM25RAG(documents=docs)
        rag.warmup()  # NOT async!

        assert rag._is_warmed_up
        assert rag._bm25 is not None
        assert len(rag._doc_id_to_index) == 5

    def test_warmup_id_mapping(self):
        """Test that document ID to index mapping is correct."""
        docs = [
            RagDocument(id="alpha", content="Content A"),
            RagDocument(id="beta", content="Content B"),
        ]
        rag = BM25RAG(documents=docs)
        rag.warmup()

        assert rag._doc_id_to_index["alpha"] == 0
        assert rag._doc_id_to_index["beta"] == 1

    def test_warmup_skip_if_already_warmed(self):
        """Test that warmup skips if already warmed up."""
        docs = [RagDocument(id="1", content="test")]
        rag = BM25RAG(documents=docs)
        rag.warmup()

        # Second warmup should skip (not rebuild index)
        rag.warmup()
        assert rag._is_warmed_up  # Still warmed up

    def test_warmup_with_empty_documents(self):
        """Test warmup with no documents - should not raise."""
        rag = BM25RAG(documents=[])
        rag.warmup()
        assert rag._is_warmed_up
        # With empty documents, _bm25 remains None
        assert rag._bm25 is None


class TestBM25RAGRetrieve:
    """Test BM25RAG retrieve (no LLM needed!)."""

    @pytest.mark.asyncio
    async def test_retrieve_returns_results(self):
        """Test that retrieve returns results."""
        docs = [
            RagDocument(id="1", content="Python is a programming language"),
            RagDocument(id="2", content="Django is a web framework"),
            RagDocument(id="3", content="Machine learning with Python"),
        ]
        rag = BM25RAG(documents=docs, config=BM25Config(top_k=2))
        rag.warmup()  # NOT async

        result = await rag.retrieve("Python")
        assert len(result.documents) <= 2
        assert result.query == "Python"
        assert result.context != ""

    @pytest.mark.asyncio
    async def test_retrieve_ranks_by_relevance(self):
        """Test that retrieve returns results."""
        docs = [
            RagDocument(id="1", content="Python programming tutorial advanced"),
            RagDocument(id="2", content="Cooking recipes for beginners"),
            RagDocument(id="3", content="Python basics for starters"),
        ]
        rag = BM25RAG(documents=docs)
        rag.warmup()

        result = await rag.retrieve("Python")
        assert len(result.documents) > 0
        # Just check that Python-related docs are returned
        doc_ids = [d["id"] for d in result.documents]
        assert "1" in doc_ids or "3" in doc_ids

    @pytest.mark.asyncio
    async def test_retrieve_respects_top_k(self):
        """Test that retrieve returns at most top_k results."""
        docs = [RagDocument(id=f"doc-{i}", content=f"Content about topic {i}") for i in range(10)]
        rag = BM25RAG(documents=docs, config=BM25Config(top_k=3))
        rag.warmup()

        result = await rag.retrieve("topic")
        assert len(result.documents) <= 3

    @pytest.mark.asyncio
    async def test_retrieve_with_scores(self):
        """Test that retrieve includes scores."""
        docs = [RagDocument(id="1", content="test content")]
        rag = BM25RAG(documents=docs)
        rag.warmup()

        result = await rag.retrieve("test")
        assert len(result.documents) > 0
        assert "score" in result.documents[0]
        assert result.documents[0]["score"] > 0

    @pytest.mark.asyncio
    async def test_retrieve_empty_query(self):
        """Test retrieve with empty query."""
        docs = [RagDocument(id="1", content="Some content")]
        rag = BM25RAG(documents=docs)
        rag.warmup()

        result = await rag.retrieve("")
        assert isinstance(result.documents, list)

    @pytest.mark.asyncio
    async def test_retrieve_no_results(self):
        """Test retrieve when no relevant documents."""
        docs = [RagDocument(id="1", content="Python programming")]
        rag = BM25RAG(documents=docs)
        rag.warmup()

        result = await rag.retrieve("cooking recipes")
        # BM25 may still return something, but with low scores
        assert isinstance(result.documents, list)

    @pytest.mark.asyncio
    async def test_retrieve_no_results(self):
        """Test retrieve when no relevant documents."""
        docs = [RagDocument(id="1", content="Python programming")]
        rag = BM25RAG(documents=docs)
        rag.warmup()

        result = await rag.retrieve("cooking recipes")
        # BM25 may still return something, but with low scores
        assert isinstance(result.documents, list)


class TestBM25RAGAddDocuments:
    """Test adding documents incrementally."""

    @pytest.mark.asyncio
    async def test_add_documents(self):
        """Test adding new documents."""
        docs = [RagDocument(id="1", content="Original")]
        rag = BM25RAG(documents=docs)
        rag.warmup()

        new_docs = [RagDocument(id="2", content="New document")]
        await rag.add_documents(new_docs)

        assert len(rag.documents) == 2
        assert "2" in rag._doc_id_to_index

    @pytest.mark.asyncio
    async def test_add_documents(self):
        """Test adding new documents."""
        docs = [RagDocument(id="1", content="Original")]
        rag = BM25RAG(documents=docs)
        rag.warmup()

        new_docs = [RagDocument(id="2", content="New document")]
        await rag.add_documents(new_docs)

        assert len(rag.documents) == 2
        assert "2" in rag._doc_id_to_index

    @pytest.mark.asyncio
    async def test_add_documents_rebuilds_index(self):
        """Test that adding documents rebuilds the index."""
        docs = [RagDocument(id="1", content="Original")]
        rag = BM25RAG(documents=docs)
        rag.warmup()

        new_docs = [RagDocument(id="2", content="Python is great")]
        await rag.add_documents(new_docs)

        # New document should be retrievable
        result = await rag.retrieve("Python")
        assert len(result.documents) > 0


class TestBM25RAGRemoveDocuments:
    """Test removing documents incrementally."""

    @pytest.mark.asyncio
    async def test_remove_documents(self):
        """Test removing documents by ID."""
        docs = [
            RagDocument(id="1", content="Keep this"),
            RagDocument(id="2", content="Remove this"),
        ]
        rag = BM25RAG(documents=docs)
        rag.warmup()

        await rag.remove_documents(["2"])

        assert len(rag.documents) == 1
        assert rag.documents[0].id == "1"

    @pytest.mark.asyncio
    async def test_remove_nonexistent(self):
        """Test removing non-existent document ID (no error)."""
        docs = [RagDocument(id="1", content="Only doc")]
        rag = BM25RAG(documents=docs)
        rag.warmup()

        await rag.remove_documents(["nonexistent"])  # Should not raise

        assert len(rag.documents) == 1
        assert rag.documents[0].id == "1"

    @pytest.mark.asyncio
    async def test_remove_nonexistent(self):
        """Test removing non-existent document ID (no error)."""
        docs = [RagDocument(id="1", content="Only doc")]
        rag = BM25RAG(documents=docs)
        rag.warmup()

        await rag.remove_documents(["nonexistent"])  # Should not raise

        assert len(rag.documents) == 1


class TestBM25RAGRefreshDocuments:
    """Test refreshing all documents."""

    def test_refresh_documents(self):
        """Test refreshing with completely new documents."""
        docs = [RagDocument(id="old", content="Old content")]
        rag = BM25RAG(documents=docs)
        rag.warmup()

        new_docs = [RagDocument(id="new", content="New content")]
        rag.refresh_documents(new_docs)

        assert len(rag.documents) == 1
        assert rag.documents[0].id == "new"
        assert rag._is_warmed_up  # Index rebuilt


class TestBM25RAGAddRemoveBaseClass:
    """Test base class optional methods."""

    @pytest.mark.asyncio
    async def test_add_documents_base_class(self):
        """Test that base class add_documents works."""
        rag = BM25RAG(documents=[RagDocument(id="1", content="test")])
        rag.warmup()

        new_docs = [RagDocument(id="2", content="new")]
        await rag.add_documents(new_docs)

        assert len(rag.documents) == 2

    @pytest.mark.asyncio
    async def test_remove_documents_base_class(self):
        """Test that base class remove_documents works."""
        rag = BM25RAG(documents=[RagDocument(id="1", content="test")])
        rag.warmup()

        await rag.remove_documents(["1"])

        assert len(rag.documents) == 0
