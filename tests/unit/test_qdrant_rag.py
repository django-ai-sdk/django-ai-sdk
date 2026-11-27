import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from django_ai_sdk.rags.qdrant_hybrid import (
    MultiQueryQdrantHybridRetriever,
    QdrantBM25HybridRAG,
    QdrantBM25HybridRAGConfig,
    dense_document_embedder,
    dense_text_embedder,
)
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack_integrations.components.embedders.fastembed import (
    FastembedSparseTextEmbedder,
    FastembedTextEmbedder,
)
from django_ai_sdk.rags.config import QdrantStorageConfig
from django_ai_sdk.rags.schemas import RagDocument
from tenacity import stop_after_delay as tenacity_stop_after_delay, wait_none


class TestQdrantRAGInit:
    def test_init_with_documents(self):
        docs = [RagDocument(id="1", content="test", title="Test")]
        rag = QdrantBM25HybridRAG(documents=docs)
        assert len(rag.documents) == 1
        assert not rag._is_warmed_up
        assert rag._cached_document_store is None

    def test_init_with_config(self):
        config = QdrantBM25HybridRAGConfig()
        docs = [RagDocument(id="1", content="test")]
        rag = QdrantBM25HybridRAG(documents=docs, config=config)
        assert rag.config == config

    def test_init_stores_documents(self):
        docs = [RagDocument(id="1", content="test", title="Test")]
        rag = QdrantBM25HybridRAG(documents=docs)
        assert len(rag.documents) == 1
        assert rag.documents[0].id == "1"


class TestQdrantRAGRemoteEmbeddings:
    def test_local_fastembed_by_default(self):
        config = QdrantBM25HybridRAGConfig()
        assert not config.remote_embeddings
        assert config.embedding_dim == 384
        assert isinstance(
            dense_text_embedder(config.dense_embedder_model, config.remote_embeddings),
            FastembedTextEmbedder,
        )

    def test_the_dimension_setting_only_applies_to_a_remote_model(self, settings):
        settings.AI_SDK_EMBEDDINGS_DIM = 4096
        assert QdrantBM25HybridRAGConfig().embedding_dim == 384
        assert QdrantBM25HybridRAGConfig(embedding_dim=768).embedding_dim == 768

    def test_setting_switches_dense_embedders_to_the_api(self, settings):
        settings.OPENAI_API_KEY = "test"
        settings.OPENAI_API_URL = "https://example.test/v1"
        settings.AI_SDK_EMBEDDINGS_MODEL = "Qwen/Qwen3-Embedding-8B"
        settings.AI_SDK_EMBEDDINGS_DIM = 4096
        config = QdrantBM25HybridRAGConfig(query_prefix="Query: ")
        assert config.embedding_dim == 4096

        doc = dense_document_embedder(
            config.dense_embedder_model, config.remote_embeddings, ["name"]
        )
        assert isinstance(doc, OpenAIDocumentEmbedder)
        assert doc.model == "Qwen/Qwen3-Embedding-8B"
        assert doc.api_base_url == "https://example.test/v1"
        assert doc.meta_fields_to_embed == ["name"]
        assert doc.prefix == ""

        retriever = MultiQueryQdrantHybridRetriever(
            document_store=None,
            dense_embedder_model=config.dense_embedder_model,
            remote_embeddings=config.remote_embeddings,
            query_prefix=config.query_prefix,
        )
        with patch.object(FastembedSparseTextEmbedder, "warm_up"):
            retriever.warm_up()
        assert isinstance(retriever._dense_embedder, OpenAITextEmbedder)
        assert retriever._dense_embedder.prefix == "Query: "


class TestQdrantRAGWarmup:
    @pytest.mark.asyncio
    async def test_warmup_creates_store(self):
        docs = [RagDocument(id=f"doc-{i}", content=f"Content {i}") for i in range(3)]
        config = QdrantBM25HybridRAGConfig()
        rag = QdrantBM25HybridRAG(documents=docs, config=config)
        await rag.warmup()

        assert rag._is_warmed_up
        assert rag._cached_document_store is not None
        assert rag._cached_document_store.count_documents() >= 3

    @pytest.mark.asyncio
    async def test_warmup_skip_if_already_warmed(self):
        docs = [RagDocument(id="1", content="test")]
        config = QdrantBM25HybridRAGConfig()
        rag = QdrantBM25HybridRAG(documents=docs, config=config)
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
        config = QdrantBM25HybridRAGConfig(chunk_size=50, chunk_overlap=10)
        rag = QdrantBM25HybridRAG(documents=docs, config=config)
        await rag.warmup()

        chunk_count = rag._cached_document_store.count_documents()
        assert chunk_count > 1

    @pytest.mark.asyncio
    async def test_warmup_with_empty_documents(self):
        config = QdrantBM25HybridRAGConfig()
        rag = QdrantBM25HybridRAG(documents=[], config=config)
        await rag.warmup()

        assert rag._is_warmed_up
        assert rag._cached_document_store is not None


class TestQdrantRAGAddDocuments:
    @pytest.mark.asyncio
    async def test_add_documents(self):
        docs = [RagDocument(id="1", content="Original", title="Orig")]
        config = QdrantBM25HybridRAGConfig()
        rag = QdrantBM25HybridRAG(documents=docs, config=config)
        await rag.warmup()

        original_count = rag._cached_document_store.count_documents()

        new_docs = [RagDocument(id="2", content="New content", title="New")]
        await rag.add_documents(new_docs)

        assert rag._cached_document_store.count_documents() >= original_count + 1

    @pytest.mark.asyncio
    async def test_add_multiple_documents(self):
        docs = [RagDocument(id="1", content="Original")]
        config = QdrantBM25HybridRAGConfig()
        rag = QdrantBM25HybridRAG(documents=docs, config=config)
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
        config = QdrantBM25HybridRAGConfig()
        rag = QdrantBM25HybridRAG(documents=docs, config=config)
        await rag.warmup()

        new_docs = [RagDocument(id="2", content="New")]
        await rag.add_documents(new_docs)

        assert rag._is_warmed_up


class TestQdrantRAGRemoveDocuments:
    @pytest.mark.asyncio
    async def test_remove_documents(self):
        docs = [
            RagDocument(id="1", content="Keep this"),
            RagDocument(id="2", content="Remove this"),
        ]
        config = QdrantBM25HybridRAGConfig()
        rag = QdrantBM25HybridRAG(documents=docs, config=config)
        await rag.warmup()
        original_count = rag._cached_document_store.count_documents()

        await rag.remove_documents(["2"])

        assert rag._cached_document_store.count_documents() <= original_count

    @pytest.mark.asyncio
    async def test_remove_multiple(self):
        docs = [
            RagDocument(id="1", content="Doc 1"),
            RagDocument(id="2", content="Doc 2"),
            RagDocument(id="3", content="Doc 3"),
        ]
        config = QdrantBM25HybridRAGConfig()
        rag = QdrantBM25HybridRAG(documents=docs, config=config)
        await rag.warmup()

        await rag.remove_documents(["1", "3"])

        assert rag._cached_document_store.count_documents() <= 2


class TestQdrantRAGRefreshDocuments:
    @pytest.mark.asyncio
    async def test_refresh_documents(self):
        docs = [RagDocument(id="old", content="old", title="Old")]
        config = QdrantBM25HybridRAGConfig()
        rag = QdrantBM25HybridRAG(documents=docs, config=config)
        await rag.warmup()

        new_docs = [RagDocument(id="new", content="new", title="New")]
        await rag.refresh_documents(new_docs)

        assert rag._cached_document_store.count_documents() >= 1

    @pytest.mark.asyncio
    async def test_refresh_reuses_store(self):
        docs = [RagDocument(id="old", content="old")]
        config = QdrantBM25HybridRAGConfig()
        rag = QdrantBM25HybridRAG(documents=docs, config=config)
        await rag.warmup()

        old_store = rag._cached_document_store

        new_docs = [RagDocument(id="new", content="new")]
        await rag.refresh_documents(new_docs)

        assert rag._cached_document_store is old_store

    @pytest.mark.asyncio
    async def test_refresh_full_rebuild(self):
        docs = [
            RagDocument(id="old1", content="old content 1"),
            RagDocument(id="old2", content="old content 2"),
        ]
        config = QdrantBM25HybridRAGConfig()
        rag = QdrantBM25HybridRAG(documents=docs, config=config)
        await rag.warmup()

        new_docs = [RagDocument(id="new1", content="new content")]
        await rag.refresh_documents(new_docs)

        assert rag._cached_document_store.count_documents() >= 1

        assert len(rag.documents) == 1
        assert rag.documents[0].id == "new1"


class TestQdrantFileLockRetry:
    @pytest.fixture(autouse=True)
    def no_retry_wait(self):
        with (
            patch(
                "django_ai_sdk.rags.qdrant_hybrid.wait_exponential",
                return_value=wait_none(),
            ),
            patch(
                "django_ai_sdk.rags.qdrant_hybrid.stop_after_delay",
                return_value=tenacity_stop_after_delay(0.1),
            ),
        ):
            yield

    @pytest.fixture
    def rag(self, tmp_path):
        storage = QdrantStorageConfig(backend="persistent", persist_path=str(tmp_path))
        config = QdrantBM25HybridRAGConfig(storage=storage)
        docs = [RagDocument(id="1", content="test", title="Test")]
        return QdrantBM25HybridRAG(documents=docs, config=config)

    @pytest.mark.asyncio
    async def test_retry_on_file_lock_contention(self, rag):
        valid_store = MagicMock()
        valid_store.count_documents.return_value = 0

        call_count = 0

        def mock_qdrant_store(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError(
                    "Storage folder "
                    f"{rag.config.storage.persist_path} is already accessed by another "
                    "instance of Qdrant client"
                )
            return valid_store

        with patch(
            "django_ai_sdk.rags.qdrant_hybrid.QdrantDocumentStore",
            side_effect=mock_qdrant_store,
        ):
            await rag.warmup()

        assert call_count == 2
        assert rag._is_warmed_up
        assert rag._cached_document_store is valid_store

    @pytest.mark.asyncio
    async def test_exhausts_retries(self, rag):
        def always_fail(*args, **kwargs):
            raise RuntimeError(
                f"Storage folder {rag.config.storage.persist_path} is already accessed by "
                "another instance of Qdrant client"
            )

        with patch(
            "django_ai_sdk.rags.qdrant_hybrid.QdrantDocumentStore",
            side_effect=always_fail,
        ):
            with pytest.raises(RuntimeError, match="already accessed by another"):
                await rag.warmup()

        assert not rag._is_warmed_up
        assert rag._cached_document_store is None

    @pytest.mark.asyncio
    async def test_non_lock_errors_propagate_immediately(self, rag):
        with patch(
            "django_ai_sdk.rags.qdrant_hybrid.QdrantDocumentStore",
            side_effect=RuntimeError("Something else went wrong"),
        ):
            with pytest.raises(RuntimeError, match="Something else went wrong"):
                await rag.warmup()

        assert not rag._is_warmed_up
        assert rag._cached_document_store is None


class TestQdrantServerCollectionRace:
    @pytest.mark.asyncio
    async def test_a_collection_another_worker_just_created_is_used(self):
        class Conflict(Exception):
            status_code = 409

        storage = QdrantStorageConfig(backend="server", location="http://qdrant:6333")
        rag = QdrantBM25HybridRAG(
            documents=[RagDocument(id="1", content="test", title="Test")],
            config=QdrantBM25HybridRAGConfig(storage=storage),
        )
        raced, ready = MagicMock(), MagicMock()
        raced.count_documents.side_effect = Conflict("Collection `memory_x` already exists!")
        ready.count_documents.return_value = 0
        with patch(
            "django_ai_sdk.rags.qdrant_hybrid.QdrantDocumentStore", side_effect=[raced, ready]
        ) as store:
            assert rag._create_document_store(recreate=True) is ready
        assert [c.kwargs["recreate_index"] for c in store.call_args_list] == [True, False]


class TestQdrantCollectionDeletedExternally:
    def test_a_missing_collection_counts_as_no_index(self):
        rag = QdrantBM25HybridRAG(documents=[], config=QdrantBM25HybridRAGConfig())
        store = rag._create_document_store()
        store.count_documents()
        store._client.delete_collection(store.index)

        assert rag._has_existing_index(store) is False


class TestQdrantRealFileLock:
    @staticmethod
    def rag(path, lock_timeout):
        storage = QdrantStorageConfig(
            backend="persistent", persist_path=str(path), lock_timeout=lock_timeout
        )
        return QdrantBM25HybridRAG(documents=[], config=QdrantBM25HybridRAGConfig(storage=storage))

    @pytest.mark.asyncio
    async def test_waits_for_the_lock_to_be_released(self, tmp_path):
        from qdrant_client import QdrantClient

        holder = QdrantClient(path=str(tmp_path))
        threading.Timer(0.5, holder.close).start()

        store = await asyncio.to_thread(self.rag(tmp_path, 10)._create_document_store)

        assert store.count_documents() == 0

    @pytest.mark.asyncio
    async def test_gives_up_as_knowledge_unavailable(self, tmp_path):
        from qdrant_client import QdrantClient

        from django_ai_sdk.errors import AiSdkError, ErrorCode
        from django_ai_sdk.rags.provider import RAGProvider

        holder = QdrantClient(path=str(tmp_path))
        agent = MagicMock()
        agent.get_rag_pipeline = AsyncMock(return_value=self.rag(tmp_path, 0.3))
        try:
            with pytest.raises(AiSdkError) as raised:
                await RAGProvider().get_rag_instance(agent, "memory-1")
        finally:
            holder.close()

        assert raised.value.code == ErrorCode.KNOWLEDGE_UNAVAILABLE


class TestQdrantIndexMismatch:
    @staticmethod
    def store(path, dim):
        storage = QdrantStorageConfig(backend="persistent", persist_path=str(path))
        config = QdrantBM25HybridRAGConfig(storage=storage, embedding_dim=dim)
        return QdrantBM25HybridRAG(documents=[], config=config)._create_document_store()

    def test_an_index_of_another_dimension_surfaces(self, tmp_path):
        self.store(tmp_path, 3).close()

        with pytest.raises(ValueError, match="vector size"):
            self.store(tmp_path, 4)


class TestQdrantMissingEmbeddings:
    @staticmethod
    def _store():
        from haystack.dataclasses import Document, SparseEmbedding
        from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

        store = QdrantDocumentStore(
            ":memory:", use_sparse_embeddings=True, embedding_dim=3, return_embedding=True
        )
        sparse = SparseEmbedding(indices=[1], values=[0.5])
        store.write_documents(
            [
                Document(id="a", content="ok", embedding=[0.0, 1.0, 0.0], sparse_embedding=sparse),
                Document(id="b", content="failed", meta={"title": "T"}, sparse_embedding=sparse),
            ]
        )
        return store

    @staticmethod
    def _embed_missing(store, vector):
        from dataclasses import replace

        seen = []

        def run(documents):
            seen.extend((d.id, d.meta.get("title")) for d in documents)
            return {"documents": [replace(d, embedding=vector) for d in documents]}

        rag = QdrantBM25HybridRAG(documents=[], config=QdrantBM25HybridRAGConfig(embedding_dim=3))
        embedder = MagicMock(run=run)
        with patch(
            "django_ai_sdk.rags.qdrant_hybrid.dense_document_embedder", return_value=embedder
        ):
            return rag._embed_missing(store), seen

    def test_only_the_chunk_without_a_dense_vector_is_embedded(self):
        store = self._store()

        result, seen = self._embed_missing(store, [1.0, 0.0, 0.0])

        assert (result, seen) == ((1, 0), [("b", "T")])
        docs = {d.id: d for d in store.filter_documents()}
        assert docs["a"].embedding == pytest.approx([0.0, 1.0, 0.0])
        assert docs["b"].embedding == pytest.approx([1.0, 0.0, 0.0])
        assert docs["b"].sparse_embedding is not None

    def test_a_failing_embedder_leaves_the_chunk_for_the_next_run(self):
        store = self._store()

        result, _ = self._embed_missing(store, None)

        assert result == (0, 1)
        assert {d.id: d.embedding for d in store.filter_documents()}["b"] is None

    def test_nothing_missing_embeds_nothing(self):
        store = self._store()
        self._embed_missing(store, [1.0, 0.0, 0.0])

        assert self._embed_missing(store, [1.0, 0.0, 0.0]) == ((0, 0), [])

    def test_an_index_of_another_dimension_is_refused(self, tmp_path):
        from haystack.dataclasses import Document, SparseEmbedding
        from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

        def store(dim):
            return QdrantDocumentStore(
                path=str(tmp_path), use_sparse_embeddings=True, embedding_dim=dim
            )

        old = store(3)
        sparse = SparseEmbedding(indices=[1], values=[0.5])
        old.write_documents([Document(id="b", content="failed", sparse_embedding=sparse)])
        old.close()

        new = store(4)
        try:
            with pytest.raises(ValueError):
                self._embed_missing(new, [1.0, 0.0, 0.0, 0.0])
        finally:
            new.close()
