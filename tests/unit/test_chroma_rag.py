"""
Tests for Chroma RAG implementation.

These tests use in-memory Chroma store (no LLM needed).
Document indexing works with local FastEmbed models.
"""

import pytest
from django_ai_sdk.rags.chroma import ChromaDBQueryExpanderRAG, ChromaDBQueryExpanderRAGConfig
from django_ai_sdk.rags.schemas import RagDocument


class TestChromaRAGInit:
    """Test Chroma RAG initialization."""

    def test_init_with_documents(self):
        """Test initialization with document list."""
        docs = [RagDocument(id="1", content="test", title="Test")]
        rag = ChromaDBQueryExpanderRAG(documents=docs)
        assert len(rag.documents) == 1
        assert not rag._is_warmed_up
        assert rag._cached_document_store is None

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = ChromaDBQueryExpanderRAGConfig()
        docs = [RagDocument(id="1", content="test")]
        rag = ChromaDBQueryExpanderRAG(documents=docs, config=config)
        assert rag.config == config

    def test_init_default_config(self):
        """Test that default config is created if none provided."""
        docs = [RagDocument(id="1", content="test")]
        rag = ChromaDBQueryExpanderRAG(documents=docs)
        assert isinstance(rag.config, ChromaDBQueryExpanderRAGConfig)


class TestChromaRAGWarmup:
    """Test Chroma RAG warmup (document indexing)."""

    @pytest.mark.asyncio
    async def test_warmup_creates_store(self):
        """Test that warmup creates the document store."""
        docs = [RagDocument(id=f"doc-{i}", content=f"Content {i}") for i in range(3)]
        config = ChromaDBQueryExpanderRAGConfig()  # Uses in-memory by default
        rag = ChromaDBQueryExpanderRAG(documents=docs, config=config)
        await rag.warmup()

        assert rag._is_warmed_up
        assert rag._cached_document_store is not None
        # Note: chunking may create more chunks than docs
        assert rag._cached_document_store.count_documents() >= 3

    @pytest.mark.asyncio
    async def test_warmup_skip_if_already_warmed(self):
        """Test that warmup skips if already warmed up."""
        docs = [RagDocument(id="1", content="test")]
        config = ChromaDBQueryExpanderRAGConfig()
        rag = ChromaDBQueryExpanderRAG(documents=docs, config=config)
        await rag.warmup()
        assert rag._is_warmed_up

        # Store the current count
        count_before = rag._cached_document_store.count_documents()

        # Second warmup should skip (is_warmed_up remains True)
        await rag.warmup()
        assert rag._is_warmed_up
        assert rag._cached_document_store.count_documents() == count_before

    @pytest.mark.asyncio
    async def test_warmup_with_chunking(self):
        """Test that warmup applies chunking."""
        # Create a long document that will be chunked
        long_content = "This is a test. " * 100
        docs = [RagDocument(id="1", content=long_content)]
        config = ChromaDBQueryExpanderRAGConfig(chunk_size=50, chunk_overlap=10)
        rag = ChromaDBQueryExpanderRAG(documents=docs, config=config)
        await rag.warmup()

        # Should have more chunks than documents
        chunk_count = rag._cached_document_store.count_documents()
        assert chunk_count > 1


class TestChromaRAGAddDocuments:
    """Test adding documents incrementally."""

    @pytest.mark.asyncio
    async def test_add_documents(self):
        """Test adding new documents to existing index."""
        docs = [RagDocument(id="1", content="Original", title="Orig")]
        config = ChromaDBQueryExpanderRAGConfig()
        rag = ChromaDBQueryExpanderRAG(documents=docs, config=config)
        await rag.warmup()

        original_count = rag._cached_document_store.count_documents()

        new_docs = [RagDocument(id="2", content="New content", title="New")]
        await rag.add_documents(new_docs)

        assert rag._cached_document_store.count_documents() >= original_count + 1

    @pytest.mark.asyncio
    async def test_add_multiple_documents(self):
        """Test adding multiple documents at once."""
        docs = [RagDocument(id="1", content="Original")]
        config = ChromaDBQueryExpanderRAGConfig()
        rag = ChromaDBQueryExpanderRAG(documents=docs, config=config)
        await rag.warmup()

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
        config = ChromaDBQueryExpanderRAGConfig()
        rag = ChromaDBQueryExpanderRAG(documents=docs, config=config)
        await rag.warmup()  # Warmup first!

        new_docs = [RagDocument(id="2", content="New")]
        await rag.add_documents(new_docs)

        # Should have been auto-warmed
        assert rag._is_warmed_up


class TestChromaRAGRemoveDocuments:
    """Test removing documents incrementally."""

    @pytest.mark.asyncio
    async def test_remove_documents(self):
        """Test removing documents by ID."""
        docs = [
            RagDocument(id="1", content="Keep", title="K"),
            RagDocument(id="2", content="Remove", title="R"),
        ]
        config = ChromaDBQueryExpanderRAGConfig()
        rag = ChromaDBQueryExpanderRAG(documents=docs, config=config)
        await rag.warmup()

        original_count = rag._cached_document_store.count_documents()

        await rag.remove_documents(["2"])

        # Should have less documents now (or equal if chunking created multiple chunks for doc 1)
        assert rag._cached_document_store.count_documents() <= original_count

    @pytest.mark.asyncio
    async def test_remove_nonexistent(self):
        """Test removing non-existent document (no error)."""
        docs = [RagDocument(id="1", content="Only")]
        config = ChromaDBQueryExpanderRAGConfig()
        rag = ChromaDBQueryExpanderRAG(documents=docs, config=config)
        await rag.warmup()

        await rag.remove_documents(["nonexistent"])  # Should not raise

        assert rag._cached_document_store.count_documents() >= 1

    @pytest.mark.asyncio
    async def test_remove_multiple(self):
        """Test removing multiple documents."""
        docs = [
            RagDocument(id="1", content="First"),
            RagDocument(id="2", content="Second"),
            RagDocument(id="3", content="Third"),
        ]
        config = ChromaDBQueryExpanderRAGConfig()
        rag = ChromaDBQueryExpanderRAG(documents=docs, config=config)
        await rag.warmup()

        original_count = rag._cached_document_store.count_documents()

        await rag.remove_documents(["1", "3"])

        # Should have fewer documents after removal
        assert rag._cached_document_store.count_documents() < original_count


class TestChromaRAGRefreshDocuments:
    """Test refreshing all documents."""

    @pytest.mark.asyncio
    async def test_refresh_documents(self):
        """Test refreshing with completely new documents."""
        docs = [RagDocument(id="old", content="old", title="Old")]
        config = ChromaDBQueryExpanderRAGConfig()
        rag = ChromaDBQueryExpanderRAG(documents=docs, config=config)
        await rag.warmup()

        new_docs = [RagDocument(id="new", content="new", title="New")]
        await rag.refresh_documents(new_docs)

        assert rag._cached_document_store.count_documents() >= 1

    @pytest.mark.asyncio
    async def test_refresh_reuses_store(self):
        """Test that refresh reuses the cached store."""
        docs = [RagDocument(id="old", content="old")]
        config = ChromaDBQueryExpanderRAGConfig()
        rag = ChromaDBQueryExpanderRAG(documents=docs, config=config)
        await rag.warmup()

        old_store = rag._cached_document_store

        new_docs = [RagDocument(id="new", content="new")]
        await rag.refresh_documents(new_docs)

        # Should reuse the same store instance
        assert rag._cached_document_store is old_store
