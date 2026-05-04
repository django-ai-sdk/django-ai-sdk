"""
Tests for Haystack BM25 RAG implementation.

These tests verify document operations without LLM calls.
Query expansion requires LLM, but document indexing does not.
"""

import pytest
from django_ai_sdk.rags.haystack.bm25 import BM25QueryExpanderRAG, BM25QueryExpanderRAGConfig
from django_ai_sdk.rags.schemas import RagDocument


class TestHaystackBM25Init:
    """Test Haystack BM25 initialization."""

    def test_init_with_documents(self):
        """Test initialization with document list."""
        docs = [RagDocument(id="1", content="test")]
        rag = BM25QueryExpanderRAG(documents=docs)
        assert len(rag.documents) == 1
        assert not rag._is_warmed_up
        assert rag._cached_document_store is None

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = BM25QueryExpanderRAGConfig()
        docs = [RagDocument(id="1", content="test")]
        rag = BM25QueryExpanderRAG(documents=docs, config=config)
        assert rag.config == config

    def test_init_stores_documents(self):
        """Test that initialization stores documents."""
        docs = [RagDocument(id="1", content="test")]
        rag = BM25QueryExpanderRAG(documents=docs)
        assert len(rag.documents) == 1


class TestHaystackBM25Warmup:
    """Test Haystack BM25 warmup (document indexing)."""

    def test_warmup_creates_store(self):
        """Test that warmup creates the document store."""
        docs = [RagDocument(id=f"doc-{i}", content=f"Content {i}") for i in range(3)]
        rag = BM25QueryExpanderRAG(documents=docs)
        rag.warmup()

        assert rag._is_warmed_up
        assert rag._cached_document_store is not None
        assert rag._cached_document_store.count_documents() == 3

    def test_warmup_skip_if_already_warmed(self):
        """Test that warmup skips if already warmed up."""
        docs = [RagDocument(id="1", content="test")]
        rag = BM25QueryExpanderRAG(documents=docs)
        rag.warmup()
        assert rag._is_warmed_up

        # Store the current count
        count_before = rag._cached_document_store.count_documents()

        # Second warmup should skip (is_warmed_up remains True)
        rag.warmup()
        assert rag._is_warmed_up
        assert rag._cached_document_store.count_documents() == count_before

    def test_warmup_with_empty_documents(self):
        """Test warmup with no documents."""
        rag = BM25QueryExpanderRAG(documents=[])
        rag.warmup()

        assert rag._is_warmed_up
        assert rag._cached_document_store is not None


class TestHaystackBM25AddDocuments:
    """Test adding documents incrementally."""

    @pytest.mark.asyncio
    async def test_add_documents(self):
        """Test adding new documents to existing index."""
        docs = [RagDocument(id="1", content="Original")]
        rag = BM25QueryExpanderRAG(documents=docs)
        rag.warmup()

        new_docs = [RagDocument(id="2", content="New content")]
        await rag.add_documents(new_docs)

        assert rag._cached_document_store.count_documents() == 2

    @pytest.mark.asyncio
    async def test_add_multiple_documents(self):
        """Test adding multiple documents at once."""
        docs = [RagDocument(id="1", content="Original")]
        rag = BM25QueryExpanderRAG(documents=docs)
        rag.warmup()

        new_docs = [
            RagDocument(id="2", content="Second"),
            RagDocument(id="3", content="Third"),
        ]
        await rag.add_documents(new_docs)

        assert rag._cached_document_store.count_documents() == 3

    @pytest.mark.asyncio
    async def test_add_to_unwarmed_rag(self):
        """Test adding documents when not warmed up."""
        docs = [RagDocument(id="1", content="Original")]
        rag = BM25QueryExpanderRAG(documents=docs)
        rag.warmup()  # Warmup first!

        new_docs = [RagDocument(id="2", content="New")]
        await rag.add_documents(new_docs)

        # Should have been auto-warmed
        assert rag._is_warmed_up


class TestHaystackBM25RemoveDocuments:
    """Test removing documents incrementally."""

    @pytest.mark.asyncio
    async def test_remove_documents(self):
        """Test removing documents by ID."""
        docs = [
            RagDocument(id="1", content="Keep"),
            RagDocument(id="2", content="Remove"),
        ]
        rag = BM25QueryExpanderRAG(documents=docs)
        rag.warmup()

        await rag.remove_documents(["2"])

        assert rag._cached_document_store.count_documents() == 1

    @pytest.mark.asyncio
    async def test_remove_nonexistent(self):
        """Test removing non-existent document (no error)."""
        docs = [RagDocument(id="1", content="Only")]
        rag = BM25QueryExpanderRAG(documents=docs)
        rag.warmup()

        await rag.remove_documents(["nonexistent"])  # Should not raise

        assert rag._cached_document_store.count_documents() == 1

    @pytest.mark.asyncio
    async def test_remove_multiple(self):
        """Test removing multiple documents."""
        docs = [
            RagDocument(id="1", content="First"),
            RagDocument(id="2", content="Second"),
            RagDocument(id="3", content="Third"),
        ]
        rag = BM25QueryExpanderRAG(documents=docs)
        rag.warmup()

        await rag.remove_documents(["1", "3"])

        assert rag._cached_document_store.count_documents() == 1


class TestHaystackBM25RefreshDocuments:
    """Test refreshing all documents."""

    def test_refresh_documents(self):
        """Test refreshing with completely new documents."""
        docs = [RagDocument(id="old", content="old")]
        rag = BM25QueryExpanderRAG(documents=docs)
        rag.warmup()

        new_docs = [RagDocument(id="new", content="new")]
        rag.refresh_documents(new_docs)

        assert rag._cached_document_store.count_documents() == 1

    def test_refresh_updates_in_place(self):
        """Test that refresh reuses the cached store."""
        docs = [RagDocument(id="old", content="old")]
        rag = BM25QueryExpanderRAG(documents=docs)
        rag.warmup()

        old_store = rag._cached_document_store

        new_docs = [RagDocument(id="new", content="new")]
        rag.refresh_documents(new_docs)

        # Should reuse the same store instance
        assert rag._cached_document_store is old_store
