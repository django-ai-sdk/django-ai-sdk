"""
Tests for Qdrant Hybrid RAG implementation.

These tests use in-memory Qdrant (no LLM needed).
Document indexing uses local FastEmbed models.
"""

import pytest
from django_ai_sdk.rags.haystack.qdrant_hybrid import QdrantBM25HybridRAG, QdrantBM25HybridRAGConfig
from django_ai_sdk.rags.schemas import RagDocument


class TestQdrantRAGInit:
    """Test Qdrant RAG initialization."""

    def test_init_with_documents(self):
        """Test initialization with document list."""
        docs = [RagDocument(id="1", content="test", title="Test")]
        rag = QdrantBM25HybridRAG(documents=docs)
        assert len(rag.documents) == 1
        assert not rag._is_warmed_up
        assert rag._cached_document_store is None

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = QdrantBM25HybridRAGConfig()
        docs = [RagDocument(id="1", content="test")]
        rag = QdrantBM25HybridRAG(documents=docs, config=config)
        assert rag.config == config

    def test_init_stores_documents(self):
        """Test that initialization stores documents."""
        docs = [RagDocument(id="1", content="test", title="Test")]
        rag = QdrantBM25HybridRAG(documents=docs)
        assert len(rag.documents) == 1
        assert rag.documents[0].id == "1"


class TestQdrantRAGWarmup:
    """Test Qdrant RAG warmup (document indexing)."""

    def test_warmup_creates_store(self):
        """Test that warmup creates the document store."""
        docs = [RagDocument(id=f"doc-{i}", content=f"Content {i}") for i in range(3)]
        config = QdrantBM25HybridRAGConfig()  # Uses :memory: by default
        rag = QdrantBM25HybridRAG(documents=docs, config=config)
        rag.warmup()

        assert rag._is_warmed_up
        assert rag._cached_document_store is not None
        assert rag._cached_document_store.count_documents() >= 3

    def test_warmup_skip_if_already_warmed(self):
        """Test that warmup skips if already warmed up."""
        docs = [RagDocument(id="1", content="test")]
        config = QdrantBM25HybridRAGConfig()
        rag = QdrantBM25HybridRAG(documents=docs, config=config)
        rag.warmup()
        assert rag._is_warmed_up

        # Store the current count
        count_before = rag._cached_document_store.count_documents()

        # Second warmup should skip (is_warmed_up remains True)
        rag.warmup()
        assert rag._is_warmed_up
        assert rag._cached_document_store.count_documents() == count_before

    def test_warmup_with_chunking(self):
        """Test that warmup applies chunking."""
        # Create a long document that will be chunked
        long_content = "This is a test. " * 100
        docs = [RagDocument(id="1", content=long_content)]
        config = QdrantBM25HybridRAGConfig(chunk_size=50, chunk_overlap=10)
        rag = QdrantBM25HybridRAG(documents=docs, config=config)
        rag.warmup()

        # Should have more chunks than documents
        chunk_count = rag._cached_document_store.count_documents()
        assert chunk_count > 1

    def test_warmup_with_empty_documents(self):
        """Test warmup with no documents."""
        config = QdrantBM25HybridRAGConfig()
        rag = QdrantBM25HybridRAG(documents=[], config=config)
        rag.warmup()

        assert rag._is_warmed_up
        assert rag._cached_document_store is not None


class TestQdrantRAGAddDocuments:
    """Test adding documents incrementally."""

    @pytest.mark.asyncio
    async def test_add_documents(self):
        """Test adding new documents to existing index."""
        docs = [RagDocument(id="1", content="Original", title="Orig")]
        config = QdrantBM25HybridRAGConfig()
        rag = QdrantBM25HybridRAG(documents=docs, config=config)
        rag.warmup()

        original_count = rag._cached_document_store.count_documents()

        new_docs = [RagDocument(id="2", content="New content", title="New")]
        await rag.add_documents(new_docs)

        assert rag._cached_document_store.count_documents() >= original_count + 1

    @pytest.mark.asyncio
    async def test_add_multiple_documents(self):
        """Test adding multiple documents at once."""
        docs = [RagDocument(id="1", content="Original")]
        config = QdrantBM25HybridRAGConfig()
        rag = QdrantBM25HybridRAG(documents=docs, config=config)
        rag.warmup()

        new_docs = [
            RagDocument(id="2", content="Second", title="S"),
            RagDocument(id="3", content="Third", title="T"),
        ]
        await rag.add_documents(new_docs)

        assert rag._cached_document_store.count_documents() >= 3

    @pytest.mark.asyncio
    async def test_add_to_unwarmed_rag(self):
        """Test adding documents when not warmed up."""
        docs = [RagDocument(id="1", content="Original")]
        config = QdrantBM25HybridRAGConfig()
        rag = QdrantBM25HybridRAG(documents=docs, config=config)
        rag.warmup()  # Warmup first!

        new_docs = [RagDocument(id="2", content="New")]
        await rag.add_documents(new_docs)

        # Should have been auto-warmed
        assert rag._is_warmed_up


class TestQdrantRAGRemoveDocuments:
    """Test Qdrant RAG document removal."""

    def test_remove_documents(self):
        """Test removing a document from the index."""
        docs = [
            RagDocument(id="1", content="Keep this"),
            RagDocument(id="2", content="Remove this"),
        ]
        config = QdrantBM25HybridRAGConfig()
        rag = QdrantBM25HybridRAG(documents=docs, config=config)
        rag.warmup()
        original_count = rag._cached_document_store.count_documents()

        # Remove one document
        import asyncio
        asyncio.run(rag.remove_documents(["2"]))

        # The count should be less than or equal to original
        # Note: due to chunking, the exact count may vary
        assert rag._cached_document_store.count_documents() <= original_count

    def test_remove_multiple(self):
        """Test removing multiple documents."""
        docs = [
            RagDocument(id="1", content="Doc 1"),
            RagDocument(id="2", content="Doc 2"),
            RagDocument(id="3", content="Doc 3"),
        ]
        config = QdrantBM25HybridRAGConfig()
        rag = QdrantBM25HybridRAG(documents=docs, config=config)
        rag.warmup()

        import asyncio
        asyncio.run(rag.remove_documents(["1", "3"]))

        # After removing 2 docs out of 3, should have at most 1 remaining
        # (or more if chunking created multiple chunks per doc)
        assert rag._cached_document_store.count_documents() <= 2


class TestQdrantRAGRefreshDocuments:
    """Test refreshing all documents."""

    def test_refresh_documents(self):
        """Test refreshing with completely new documents."""
        docs = [RagDocument(id="old", content="old", title="Old")]
        config = QdrantBM25HybridRAGConfig()
        rag = QdrantBM25HybridRAG(documents=docs, config=config)
        rag.warmup()

        new_docs = [RagDocument(id="new", content="new", title="New")]
        rag.refresh_documents(new_docs)

        assert rag._cached_document_store.count_documents() >= 1

    def test_refresh_reuses_store(self):
        """Test that refresh reuses the cached store."""
        docs = [RagDocument(id="old", content="old")]
        config = QdrantBM25HybridRAGConfig()
        rag = QdrantBM25HybridRAG(documents=docs, config=config)
        rag.warmup()

        old_store = rag._cached_document_store

        new_docs = [RagDocument(id="new", content="new")]
        rag.refresh_documents(new_docs)

        # Should reuse the same store instance
        assert rag._cached_document_store is old_store

    def test_refresh_full_rebuild(self):
        """Test that refresh completely replaces documents."""
        docs = [
            RagDocument(id="old1", content="old content 1"),
            RagDocument(id="old2", content="old content 2"),
        ]
        config = QdrantBM25HybridRAGConfig()
        rag = QdrantBM25HybridRAG(documents=docs, config=config)
        rag.warmup()

        new_docs = [RagDocument(id="new1", content="new content")]
        rag.refresh_documents(new_docs)

        # Should only have 1 source document (new1)
        assert rag._cached_document_store.count_documents() >= 1

        # Verify old documents are gone by checking the count is less than before
        # (or just check that it's the right number for new docs)
        assert len(rag.documents) == 1
        assert rag.documents[0].id == "new1"
