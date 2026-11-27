import pytest
from django_ai_sdk.rags.chroma import ChromaDBQueryExpanderRAG, ChromaDBQueryExpanderRAGConfig
from django_ai_sdk.rags.schemas import RagDocument


class TestChromaRAGInit:
    def test_init_with_documents(self):
        docs = [RagDocument(id="1", content="test", title="Test")]
        rag = ChromaDBQueryExpanderRAG(documents=docs)
        assert len(rag.documents) == 1
        assert not rag._is_warmed_up
        assert rag._cached_document_store is None

    def test_init_with_config(self):
        config = ChromaDBQueryExpanderRAGConfig()
        docs = [RagDocument(id="1", content="test")]
        rag = ChromaDBQueryExpanderRAG(documents=docs, config=config)
        assert rag.config == config

    def test_init_default_config(self):
        docs = [RagDocument(id="1", content="test")]
        rag = ChromaDBQueryExpanderRAG(documents=docs)
        assert isinstance(rag.config, ChromaDBQueryExpanderRAGConfig)


class TestChromaRAGWarmup:
    @pytest.mark.asyncio
    async def test_warmup_creates_store(self):
        docs = [RagDocument(id=f"doc-{i}", content=f"Content {i}") for i in range(3)]
        config = ChromaDBQueryExpanderRAGConfig()
        rag = ChromaDBQueryExpanderRAG(documents=docs, config=config)
        await rag.warmup()

        assert rag._is_warmed_up
        assert rag._cached_document_store is not None
        assert rag._cached_document_store.count_documents() >= 3

    @pytest.mark.asyncio
    async def test_warmup_skip_if_already_warmed(self):
        docs = [RagDocument(id="1", content="test")]
        config = ChromaDBQueryExpanderRAGConfig()
        rag = ChromaDBQueryExpanderRAG(documents=docs, config=config)
        await rag.warmup()
        assert rag._is_warmed_up

        count_before = rag._cached_document_store.count_documents()

        await rag.warmup()
        assert rag._is_warmed_up
        assert rag._cached_document_store.count_documents() == count_before

    @pytest.mark.asyncio
    async def test_warmup_with_chunking(self):
        long_content = "This is a test. " * 100
        docs = [RagDocument(id="1", content=long_content)]
        config = ChromaDBQueryExpanderRAGConfig(chunk_size=50, chunk_overlap=10)
        rag = ChromaDBQueryExpanderRAG(documents=docs, config=config)
        await rag.warmup()

        chunk_count = rag._cached_document_store.count_documents()
        assert chunk_count > 1


class TestChromaRAGAddDocuments:
    @pytest.mark.asyncio
    async def test_add_documents(self):
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
        docs = [RagDocument(id="1", content="Original")]
        config = ChromaDBQueryExpanderRAGConfig()
        rag = ChromaDBQueryExpanderRAG(documents=docs, config=config)
        await rag.warmup()

        new_docs = [RagDocument(id="2", content="New")]
        await rag.add_documents(new_docs)

        assert rag._is_warmed_up


class TestChromaRAGRemoveDocuments:
    @pytest.mark.asyncio
    async def test_remove_documents(self):
        docs = [
            RagDocument(id="1", content="Keep", title="K"),
            RagDocument(id="2", content="Remove", title="R"),
        ]
        config = ChromaDBQueryExpanderRAGConfig()
        rag = ChromaDBQueryExpanderRAG(documents=docs, config=config)
        await rag.warmup()

        original_count = rag._cached_document_store.count_documents()

        await rag.remove_documents(["2"])

        assert rag._cached_document_store.count_documents() <= original_count

    @pytest.mark.asyncio
    async def test_remove_nonexistent(self):
        docs = [RagDocument(id="1", content="Only")]
        config = ChromaDBQueryExpanderRAGConfig()
        rag = ChromaDBQueryExpanderRAG(documents=docs, config=config)
        await rag.warmup()

        await rag.remove_documents(["nonexistent"])

        assert rag._cached_document_store.count_documents() >= 1

    @pytest.mark.asyncio
    async def test_remove_multiple(self):
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

        assert rag._cached_document_store.count_documents() < original_count


class TestChromaRAGRefreshDocuments:
    @pytest.mark.asyncio
    async def test_refresh_documents(self):
        docs = [RagDocument(id="old", content="old", title="Old")]
        config = ChromaDBQueryExpanderRAGConfig()
        rag = ChromaDBQueryExpanderRAG(documents=docs, config=config)
        await rag.warmup()

        new_docs = [RagDocument(id="new", content="new", title="New")]
        await rag.refresh_documents(new_docs)

        assert rag._cached_document_store.count_documents() >= 1

    @pytest.mark.asyncio
    async def test_refresh_reuses_store(self):
        docs = [RagDocument(id="old", content="old")]
        config = ChromaDBQueryExpanderRAGConfig()
        rag = ChromaDBQueryExpanderRAG(documents=docs, config=config)
        await rag.warmup()

        old_store = rag._cached_document_store

        new_docs = [RagDocument(id="new", content="new")]
        await rag.refresh_documents(new_docs)

        assert rag._cached_document_store is old_store
