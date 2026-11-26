from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any

from haystack import Pipeline, component
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.preprocessors import RecursiveDocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.core.super_component import SuperComponent
from haystack.dataclasses import Document as HaystackDocument
from haystack.document_stores.types import DuplicatePolicy
from haystack.tools import ComponentTool
from haystack_integrations.components.embedders.fastembed import (
    FastembedDocumentEmbedder,
    FastembedSparseDocumentEmbedder,
    FastembedSparseTextEmbedder,
    FastembedTextEmbedder,
)
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.document_stores.qdrant.document_store import QdrantStoreError
from pydantic import Field
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_delay,
    wait_exponential,
)

from django_ai_sdk.generators.base import build_kwargs, resolve_secret
from django_ai_sdk.logger import get_logger
from django_ai_sdk.rags.base import RAGBase, RAGConfig
from django_ai_sdk.rags.components import CANDIDATES_PER_QUERY, MultiQueryDeduplicationMixin
from django_ai_sdk.rags.config import QdrantStorageConfig
from django_ai_sdk.rags.utils import to_document
from django_ai_sdk.utils import resolve_setting

if TYPE_CHECKING:
    from django_ai_sdk.rags.schemas import RagDocument

logger = get_logger(__name__)

DEFAULT_DENSE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _created_meanwhile(exc: BaseException) -> bool:
    """Qdrant's 409 for a collection another worker created after this one looked."""
    return getattr(exc, "status_code", None) == 409 and "already exists" in str(exc)


def _collection_missing(exc: QdrantStoreError) -> bool:
    # count_documents wraps a local client's ValueError or a server's 404.
    cause = exc.__cause__
    return isinstance(cause, ValueError) or getattr(cause, "status_code", None) == 404


def _openai_kwargs(**kwargs: Any) -> dict[str, Any]:
    return build_kwargs(
        {
            "api_key": resolve_secret("OPENAI_API_KEY"),
            "api_base_url": resolve_setting("OPENAI_API_URL"),
        },
        kwargs,
    )


def dense_document_embedder(
    model: str, remote: bool, meta_fields_to_embed: list[str]
) -> FastembedDocumentEmbedder | OpenAIDocumentEmbedder:
    if remote:
        return OpenAIDocumentEmbedder(
            **_openai_kwargs(model=model, meta_fields_to_embed=meta_fields_to_embed)
        )
    return FastembedDocumentEmbedder(model=model, meta_fields_to_embed=meta_fields_to_embed)


def dense_text_embedder(
    model: str, remote: bool, prefix: str = ""
) -> FastembedTextEmbedder | OpenAITextEmbedder:
    if remote:
        return OpenAITextEmbedder(**_openai_kwargs(model=model, prefix=prefix))
    return FastembedTextEmbedder(model=model, prefix=prefix)


class QdrantBM25HybridRAGConfig(RAGConfig):
    sparse_embedder_model: str = Field(
        default="Qdrant/bm42-all-minilm-l6-v2-attentions",
    )
    dense_embedder_model: str = Field(
        default_factory=lambda: resolve_setting("AI_SDK_EMBEDDINGS_MODEL") or DEFAULT_DENSE_MODEL,
    )
    remote_embeddings: bool = Field(
        default_factory=lambda: bool(resolve_setting("AI_SDK_EMBEDDINGS_MODEL")),
    )
    embedding_dim: int = Field(
        default_factory=lambda: (
            resolve_setting("AI_SDK_EMBEDDINGS_DIM", 384)
            if resolve_setting("AI_SDK_EMBEDDINGS_MODEL")
            else 384
        ),
        ge=1,
    )
    query_prefix: str = ""
    chunk_size: int = Field(default=500, ge=1)
    chunk_overlap: int = Field(default=150, ge=0)
    meta_fields_to_embed: list[str] = Field(default=["title"])
    storage: QdrantStorageConfig = Field(default_factory=QdrantStorageConfig)


class QdrantBM25HybridRAG(RAGBase[QdrantBM25HybridRAGConfig]):
    def __init__(
        self,
        documents: list[RagDocument],
        config: QdrantBM25HybridRAGConfig | None = None,
    ) -> None:
        self.config: QdrantBM25HybridRAGConfig = config or QdrantBM25HybridRAGConfig()
        self.documents: list[RagDocument] = documents
        self._cached_document_store: QdrantDocumentStore | None = None
        self._is_warmed_up = False
        logger.info(f"QdrantBM25HybridRAG initialized with {len(documents)} documents")
        for i, doc in enumerate(documents):
            title = doc.title or (doc.metadata.get("title") if doc.metadata else None) or "N/A"
            logger.debug(
                f"  Document {i + 1}: id={doc.id}, title='{title}', content_len={len(doc.content)}"
            )

    def _convert_documents(self) -> list[HaystackDocument]:
        return [to_document(doc) for doc in self.documents]

    def _create_document_store(self, recreate: bool = False) -> QdrantDocumentStore:
        storage = self.config.storage
        extra = storage.extra

        if storage.is_server:

            def connect(recreate: bool) -> QdrantDocumentStore:
                store = QdrantDocumentStore(
                    location=storage.location,
                    recreate_index=recreate,
                    return_embedding=True,
                    use_sparse_embeddings=True,
                    embedding_dim=self.config.embedding_dim,
                    similarity=storage.similarity,
                    **extra,
                )
                store.count_documents()
                return store

            try:
                return connect(recreate)
            except Exception as exc:
                if not _created_meanwhile(exc):
                    raise
            return connect(False)

        if storage.is_persistent and storage.persist_path:
            os.makedirs(storage.persist_path, exist_ok=True)

            extra.setdefault("index", "documents")

            for attempt in Retrying(
                stop=stop_after_delay(storage.lock_timeout),
                wait=wait_exponential(multiplier=0.25, max=2),
                retry=retry_if_exception(
                    lambda e: isinstance(e, RuntimeError) and "already accessed" in str(e)
                ),
                reraise=True,
                before_sleep=lambda rs: logger.warning(
                    "Qdrant store at {} locked by another process, retrying in {:.1f}s",
                    storage.persist_path,
                    rs.next_action.sleep if rs.next_action else 0,
                ),
            ):
                with attempt:
                    store = QdrantDocumentStore(
                        path=storage.persist_path,
                        recreate_index=recreate,
                        return_embedding=True,
                        use_sparse_embeddings=True,
                        embedding_dim=self.config.embedding_dim,
                        similarity=storage.similarity,
                        **extra,
                    )
                    store.count_documents()
                    return store
            raise RuntimeError("Failed to create QdrantDocumentStore after retries")
        else:
            return QdrantDocumentStore(
                ":memory:",
                recreate_index=True,
                return_embedding=True,
                use_sparse_embeddings=True,
                embedding_dim=self.config.embedding_dim,
            )

    async def embed_missing(self) -> tuple[int, int]:
        store = self._cached_document_store or await asyncio.to_thread(self._create_document_store)
        try:
            return await asyncio.to_thread(self._embed_missing, store)
        finally:
            if store is not self._cached_document_store:
                store.close()

    def _embed_missing(self, store: QdrantDocumentStore) -> tuple[int, int]:
        missing = [doc for doc in store.filter_documents() if doc.embedding is None]
        if not missing:
            return 0, 0
        embedder = dense_document_embedder(
            self.config.dense_embedder_model,
            self.config.remote_embeddings,
            self.config.meta_fields_to_embed,
        )
        embedder.warm_up()
        embedded = [
            doc for doc in embedder.run(documents=missing)["documents"] if doc.embedding is not None
        ]
        store.write_documents(embedded, policy=DuplicatePolicy.OVERWRITE)
        return len(embedded), len(missing) - len(embedded)

    def _has_existing_index(self, document_store: QdrantDocumentStore) -> bool:
        try:
            return document_store.count_documents() > 0
        except ValueError:
            return False
        except QdrantStoreError as exc:
            if _collection_missing(exc):
                return False
            raise

    async def add_documents(self, documents: list[RagDocument]) -> None:
        if self._cached_document_store is None:
            logger.warning("No document store available, cannot add documents")
            return

        haystack_docs = [to_document(doc) for doc in documents]
        await self._index_documents(haystack_docs, self._cached_document_store)
        logger.info(f"Added {len(documents)} documents to Qdrant index")

    async def _index_documents(self, documents: list, document_store: QdrantDocumentStore) -> None:
        for doc in documents:
            if "doc_id" not in doc.meta:
                doc.meta["doc_id"] = doc.id

        indexing_pipeline = Pipeline()
        indexing_pipeline.add_component(
            "splitter",
            RecursiveDocumentSplitter(
                split_length=self.config.chunk_size,
                split_overlap=self.config.chunk_overlap,
                separators=["\n\n", "\n", ".", " "],
            ),
        )
        indexing_pipeline.add_component(
            "sparse_doc_embedder",
            FastembedSparseDocumentEmbedder(
                model=self.config.sparse_embedder_model,
                meta_fields_to_embed=self.config.meta_fields_to_embed,
            ),
        )
        indexing_pipeline.add_component(
            "dense_doc_embedder",
            dense_document_embedder(
                self.config.dense_embedder_model,
                self.config.remote_embeddings,
                self.config.meta_fields_to_embed,
            ),
        )
        indexing_pipeline.add_component(
            "writer",
            DocumentWriter(
                document_store=document_store,
                policy=DuplicatePolicy.OVERWRITE,
            ),
        )

        indexing_pipeline.connect("splitter", "sparse_doc_embedder")
        indexing_pipeline.connect("sparse_doc_embedder", "dense_doc_embedder")
        indexing_pipeline.connect("dense_doc_embedder", "writer")

        await asyncio.to_thread(indexing_pipeline.run, {"documents": documents})

    async def remove_documents(self, document_ids: list[str]) -> None:
        if self._cached_document_store is None:
            logger.warning("No document store available, cannot remove documents")
            return

        from qdrant_client.http.models import FieldCondition, Filter, MatchAny

        filter_obj = Filter(
            should=[FieldCondition(key="meta.doc_id", match=MatchAny(any=document_ids))]
        )

        self._cached_document_store.delete_by_filter(filters=filter_obj)  # ty: ignore[invalid-argument-type]
        logger.info(f"Removed {len(document_ids)} documents from Qdrant index")

    async def warmup(self, force_rebuild: bool = False) -> None:
        if self._is_warmed_up and not force_rebuild:
            logger.debug("QdrantBM25HybridRAG already warmed up, skipping")
            return

        if force_rebuild:
            logger.info("Force rebuild requested, resetting Qdrant index")
            self._is_warmed_up = False

        logger.debug("Warming up QdrantBM25HybridRAG - building indexed document store")
        logger.info(
            f"[warmup] force_rebuild={force_rebuild}, source_documents={len(self.documents)}"
        )

        storage = self.config.storage
        document_store = await asyncio.to_thread(
            self._create_document_store, recreate=force_rebuild
        )

        if (
            not force_rebuild
            and (storage.is_persistent or storage.is_server)
            and self._has_existing_index(document_store)
        ):
            existing_count = document_store.count_documents()
            self._cached_document_store = document_store
            self._is_warmed_up = True
            if storage.is_server:
                collection = storage.extra.get("index", "default")
                logger.info(
                    f"Using existing Qdrant index at {storage.location}/{collection} with {existing_count} chunks"
                )
            else:
                logger.info(
                    f"Using existing Qdrant index from {storage.persist_path} with {existing_count} chunks"
                )
            return

        if not force_rebuild and storage.is_server and not self._has_existing_index(document_store):
            document_store = await asyncio.to_thread(self._create_document_store, recreate=True)

        if storage.is_server:
            collection = storage.extra.get("index", "default")
            logger.info(f"Creating new Qdrant index at {storage.location}/{collection}")
        elif storage.is_persistent:
            logger.info(
                f"Creating new Qdrant index for persistent storage at {storage.persist_path}"
            )
        else:
            logger.info("Creating in-memory Qdrant index")

        haystack_docs = self._convert_documents()
        logger.info(f"[warmup] Converted {len(haystack_docs)} HaystackDocuments")

        logger.debug(
            f"Writing {len(haystack_docs)} documents to Qdrant with chunking (size={self.config.chunk_size}, overlap={self.config.chunk_overlap})"
        )
        await self._index_documents(haystack_docs, document_store)

        indexed_count = document_store.count_documents()
        self._cached_document_store = document_store
        self._is_warmed_up = True
        logger.info(
            f"QdrantBM25HybridRAG warmup complete: {len(self.documents)} source docs → {indexed_count} chunks indexed"
        )

    async def build_pipeline(self) -> Pipeline:
        logger.debug("Building Qdrant Hybrid RAG query pipeline")

        if self._cached_document_store is not None:
            document_store = self._cached_document_store
            if not self._has_existing_index(document_store):
                logger.warning("Qdrant collection was deleted externally, rebuilding index...")
                self._cached_document_store = None
                self._is_warmed_up = False
                await self.warmup()
                document_store = self._cached_document_store
        else:
            document_store = await asyncio.to_thread(self._create_document_store, recreate=False)

            if not self._has_existing_index(document_store):
                haystack_docs = self._convert_documents()
                await self._index_documents(haystack_docs, document_store)

        query_expander = self.get_query_expander()

        query_pipeline = Pipeline()
        query_pipeline.add_component("expander", query_expander)
        query_pipeline.add_component(
            "retriever",
            MultiQueryQdrantHybridRetriever(
                document_store=document_store,
                top_k=self.config.top_k,
                min_score=self.config.min_score,
                sparse_embedder_model=self.config.sparse_embedder_model,
                dense_embedder_model=self.config.dense_embedder_model,
                remote_embeddings=self.config.remote_embeddings,
                query_prefix=self.config.query_prefix,
            ),
        )

        query_pipeline.connect("expander.queries", "retriever.queries")

        logger.debug("Qdrant Hybrid RAG pipeline built")
        return query_pipeline

    async def refresh_documents(self, documents: list[RagDocument]) -> None:
        """Update the indexed documents without releasing the Qdrant file lock.

        Safe to call on a warmed-up instance. Reuses the open store: Qdrant's local
        file lock is non-blocking, so a second client on the same folder raises.
        """
        self.documents = documents
        logger.info(f"[refresh_documents] Refreshing Qdrant index with {len(documents)} documents")

        if self._cached_document_store is None:
            await self.warmup(force_rebuild=True)
            return

        document_store = self._cached_document_store

        existing_docs = document_store.filter_documents()
        if existing_docs:
            document_store.delete_documents(document_ids=[doc.id for doc in existing_docs])

        haystack_docs = self._convert_documents()
        await self._index_documents(haystack_docs, document_store)

        indexed_count = document_store.count_documents()
        logger.info(
            f"[refresh_documents] Done: {len(documents)} source docs → {indexed_count} chunks"
        )

    async def as_tool(self) -> ComponentTool:
        if self.needs_warmup:
            logger.debug("RAG needs warmup before creating tool, warming up now")
            await self.warmup()

        logger.debug("Creating Qdrant Hybrid RAG pipeline as ComponentTool")
        pipeline = await self.build_pipeline()

        rag_super = SuperComponent(
            pipeline=pipeline,
            input_mapping={"query": ["expander.query", "retriever.query"]},
            output_mapping={"retriever.documents": "documents"},
        )

        logger.debug("Qdrant Hybrid RAG ComponentTool created successfully")

        return ComponentTool(
            component=rag_super,
            name="hybrid_rag_tool",
            description="Retrieves relevant documents using hybrid search with query expansion.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to retrieve relevant documents.",
                    }
                },
                "required": ["query"],
            },
        )

    async def get_chunk(self, chunk_id: str) -> str | None:
        """Only uses the already-open cached store — never opens a second connection,
        which would conflict with the exclusive file lock held during RAG warmup.
        """
        if self._cached_document_store is None:
            return None
        docs = self._cached_document_store.get_documents_by_id([chunk_id])
        return docs[0].content if docs else None


@component
class MultiQueryQdrantHybridRetriever(MultiQueryDeduplicationMixin):
    def __init__(
        self,
        document_store: Any,
        top_k: int = 3,
        min_score: float | None = None,
        sparse_embedder_model: str = "Qdrant/bm42-all-minilm-l6-v2-attentions",
        dense_embedder_model: str = DEFAULT_DENSE_MODEL,
        remote_embeddings: bool = False,
        query_prefix: str = "",
    ) -> None:
        self.document_store = document_store
        self.top_k = top_k
        self.min_score = min_score
        self.sparse_embedder_model = sparse_embedder_model
        self.dense_embedder_model = dense_embedder_model
        self.remote_embeddings = remote_embeddings
        self.query_prefix = query_prefix
        self._sparse_embedder: Any | None = None
        self._dense_embedder: Any | None = None

    def warm_up(self) -> None:

        self._sparse_embedder = FastembedSparseTextEmbedder(
            model=self.sparse_embedder_model,
        )
        self._sparse_embedder.warm_up()

        self._dense_embedder = dense_text_embedder(
            self.dense_embedder_model, self.remote_embeddings, self.query_prefix
        )
        self._dense_embedder.warm_up()

    @component.output_types(documents=list[HaystackDocument])
    def run(
        self, queries: str | list[str], top_k: int | None = None, query: str | None = None
    ) -> dict[str, list[HaystackDocument]]:
        """Args:
            queries: Single query string or list of query strings
            top_k: Maximum documents to return
            query: The user's query before expansion (its words count verbatim)

        Returns:
            Dict with deduplicated documents sorted by score
        """
        if self._sparse_embedder is None:
            self.warm_up()

        if self._sparse_embedder is None or self._dense_embedder is None:
            raise ValueError("Embedders not initialized after warm_up()")

        if isinstance(queries, str):
            queries = [queries]

        k = top_k if top_k is not None else self.top_k

        results: list[dict[str, list[HaystackDocument]]] = []
        for q in queries:
            sparse_result = self._sparse_embedder.run(text=q)
            dense_result = self._dense_embedder.run(text=q)
            retriever = QdrantHybridRetriever(
                document_store=self.document_store,
                score_threshold=self.min_score,
            )
            result = retriever.run(
                query_sparse_embedding=sparse_result["sparse_embedding"],
                query_embedding=dense_result["embedding"],
                top_k=k * CANDIDATES_PER_QUERY,
            )
            results.append(result)

        docs = self.fuse_and_rank(results, queries, k, self.min_score, query=query)
        return {"documents": docs}

    @component.output_types(documents=list[HaystackDocument])
    async def run_async(
        self, queries: str | list[str], top_k: int | None = None, query: str | None = None
    ) -> dict[str, list[HaystackDocument]]:
        if isinstance(queries, str):
            queries = [queries]
        k = top_k if top_k is not None else self.top_k
        if self._sparse_embedder is None:
            await asyncio.to_thread(self.warm_up)
        if self._sparse_embedder is None or self._dense_embedder is None:
            raise ValueError("Embedders not initialized after warm_up()")
        results: list[dict[str, list[HaystackDocument]]] = []
        for q in queries:
            sparse_result = await asyncio.to_thread(self._sparse_embedder.run, text=q)
            dense_result = await asyncio.to_thread(self._dense_embedder.run, text=q)
            retriever = QdrantHybridRetriever(
                document_store=self.document_store,
                score_threshold=self.min_score,
            )
            result = await asyncio.to_thread(
                retriever.run,
                query_sparse_embedding=sparse_result["sparse_embedding"],
                query_embedding=dense_result["embedding"],
                top_k=k * CANDIDATES_PER_QUERY,
            )
            results.append(result)
        docs = self.fuse_and_rank(results, queries, k, self.min_score, query=query)
        return {"documents": docs}
