from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any

from haystack import Pipeline, component
from haystack.components.preprocessors import RecursiveDocumentSplitter
from haystack.components.query import QueryExpander
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
from pydantic import Field
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_delay,
    wait_exponential,
)

from django_ai_sdk.generators import openai_chat
from django_ai_sdk.logger import get_logger
from django_ai_sdk.rags.base import RAGBase, RAGConfig
from django_ai_sdk.rags.components import MultiQueryDeduplicationMixin
from django_ai_sdk.rags.config import QdrantStorageConfig
from django_ai_sdk.rags.utils import to_document

if TYPE_CHECKING:
    from django_ai_sdk.rags.schemas import RagDocument

logger = get_logger(__name__)


class QdrantBM25HybridRAGConfig(RAGConfig):
    """Configuration for Qdrant Hybrid RAG (BM42 Sparse + Dense)."""

    sparse_embedder_model: str = Field(
        default="Qdrant/bm42-all-minilm-l6-v2-attentions",
    )
    dense_embedder_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    embedding_dim: int = Field(default=384, ge=1)
    chunk_size: int = Field(default=500, ge=1)
    chunk_overlap: int = Field(default=150, ge=0)
    meta_fields_to_embed: list[str] = Field(default=["title"])
    storage: QdrantStorageConfig = Field(default_factory=QdrantStorageConfig)


class QdrantBM25HybridRAG(RAGBase[QdrantBM25HybridRAGConfig]):
    """RAG implementation using Qdrant with Hybrid retrieval + Query Expansion."""

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
        """Convert RagDocuments to HaystackDocuments for internal use."""
        return [to_document(doc) for doc in self.documents]

    def _create_document_store(self, recreate: bool = False) -> QdrantDocumentStore:
        """Create document store based on persistence configuration."""
        storage = self.config.storage
        extra = storage.extra

        if storage.is_server:
            return QdrantDocumentStore(
                location=storage.location,
                recreate_index=recreate,
                return_embedding=True,
                use_sparse_embeddings=True,
                embedding_dim=self.config.embedding_dim,
                similarity=storage.similarity,
                **extra,
            )

        if storage.is_persistent and storage.persist_path:
            os.makedirs(storage.persist_path, exist_ok=True)

            extra.setdefault("index", "documents")

            for attempt in Retrying(
                stop=stop_after_delay(300),
                wait=wait_exponential(multiplier=1, min=1, max=30),
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
                    return QdrantDocumentStore(
                        path=storage.persist_path,
                        recreate_index=recreate,
                        return_embedding=True,
                        use_sparse_embeddings=True,
                        embedding_dim=self.config.embedding_dim,
                        similarity=storage.similarity,
                        **extra,
                    )
            raise RuntimeError("Failed to create QdrantDocumentStore after retries")
        else:
            return QdrantDocumentStore(
                ":memory:",
                recreate_index=True,
                return_embedding=True,
                use_sparse_embeddings=True,
                embedding_dim=self.config.embedding_dim,
            )

    def _has_existing_index(self, document_store: QdrantDocumentStore) -> bool:
        """Check if document store already has indexed documents."""
        try:
            return document_store.count_documents() > 0
        except ValueError:
            return False

    async def add_documents(self, documents: list[RagDocument]) -> None:
        """Add documents to the existing Qdrant index."""
        if self._cached_document_store is None:
            logger.warning("No document store available, cannot add documents")
            return

        haystack_docs = [to_document(doc) for doc in documents]
        await self._index_documents(haystack_docs, self._cached_document_store)
        logger.info(f"Added {len(documents)} documents to Qdrant index")

    async def _index_documents(self, documents: list, document_store: QdrantDocumentStore) -> None:
        """Index documents with chunking and embedding."""

        # Add original doc_id to metadata so we can delete by it later
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
            FastembedDocumentEmbedder(
                model=self.config.dense_embedder_model,
                meta_fields_to_embed=self.config.meta_fields_to_embed,
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

        # Run sync pipeline in thread pool so embeddings don't block the event loop
        await asyncio.to_thread(indexing_pipeline.run, {"documents": documents})

    async def remove_documents(self, document_ids: list[str]) -> None:
        """Remove documents from the Qdrant index."""
        if self._cached_document_store is None:
            logger.warning("No document store available, cannot remove documents")
            return

        try:
            # Use delete_by_filter with metadata filtering
            # Build filter for doc_id field in metadata
            from qdrant_client.http.models import FieldCondition, Filter, MatchAny

            filter_obj = Filter(
                should=[FieldCondition(key="meta.doc_id", match=MatchAny(any=document_ids))]
            )

            self._cached_document_store.delete_by_filter(filters=filter_obj)  # ty: ignore[invalid-argument-type]
            logger.info(f"Removed {len(document_ids)} documents from Qdrant index")
        except Exception as e:
            logger.error(f"Failed to remove documents: {e}")

    async def warmup(self, force_rebuild: bool = False) -> None:
        """
        Build or load indexed document store.

        Args:
            force_rebuild: If True, clears existing index and rebuilds from scratch.
                          This will delete all documents in the persistent storage.
        """
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
        document_store = self._create_document_store(recreate=force_rebuild)

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

        # If server collection was deleted externally, recreate the store
        # so _index_documents has a collection to write into.
        if not force_rebuild and storage.is_server and not self._has_existing_index(document_store):
            document_store = self._create_document_store(recreate=True)

        if storage.is_server:
            collection = storage.extra.get("index", "default")
            logger.info(f"Creating new Qdrant index at {storage.location}/{collection}")
        elif storage.is_persistent:
            logger.info(
                f"Creating new Qdrant index for persistent storage at {storage.persist_path}"
            )
        else:
            logger.info("Creating in-memory Qdrant index")

        # Convert RagDocuments to HaystackDocuments
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
            document_store = self._create_document_store(recreate=False)

            if not self._has_existing_index(document_store):
                haystack_docs = self._convert_documents()
                await self._index_documents(haystack_docs, document_store)

        expander_generator = openai_chat(model=self.config.expander_model)

        query_expander = QueryExpander(
            chat_generator=expander_generator,
            n_expansions=self.config.n_expansions,
            prompt_template=self.config.expander_prompt,
        )

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
            ),
        )

        query_pipeline.connect("expander.queries", "retriever.queries")

        logger.debug("Qdrant Hybrid RAG pipeline built")
        return query_pipeline

    async def refresh_documents(self, documents: list[RagDocument]) -> None:
        """
        Update the indexed documents without releasing the Qdrant file lock.

        Replaces self.documents with the new list and re-indexes all of them
        into the already-open document store using OVERWRITE policy.
        Safe to call on a warmed-up instance — avoids creating a second
        QdrantClient on the same storage folder (which would deadlock).
        """
        self.documents = documents
        logger.info(f"[refresh_documents] Refreshing Qdrant index with {len(documents)} documents")

        if self._cached_document_store is None:
            # Not yet warmed up — do a full warmup with force_rebuild
            await self.warmup(force_rebuild=True)
            return

        document_store = self._cached_document_store

        # Wipe and rewrite all chunks so deleted/updated docs don't linger
        existing_docs = document_store.filter_documents()
        if existing_docs:
            document_store.delete_documents(document_ids=[doc.id for doc in existing_docs])

        # Convert and index using shared helper
        haystack_docs = self._convert_documents()
        await self._index_documents(haystack_docs, document_store)

        indexed_count = document_store.count_documents()
        logger.info(
            f"[refresh_documents] Done: {len(documents)} source docs → {indexed_count} chunks"
        )

    async def as_tool(self) -> ComponentTool:
        """Return the RAG pipeline as a ComponentTool."""
        if self.needs_warmup:
            logger.debug("RAG needs warmup before creating tool, warming up now")
            await self.warmup()

        logger.debug("Creating Qdrant Hybrid RAG pipeline as ComponentTool")
        pipeline = await self.build_pipeline()

        rag_super = SuperComponent(
            pipeline=pipeline,
            input_mapping={"query": ["expander.query"]},
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
        """Fetch a single chunk from the Qdrant store by its Haystack document ID.

        Only uses the already-open cached store — never opens a second connection,
        which would conflict with the exclusive file lock held during RAG warmup.
        """
        if self._cached_document_store is None:
            return None
        docs = self._cached_document_store.get_documents_by_id([chunk_id])
        return docs[0].content if docs else None


@component
class MultiQueryQdrantHybridRetriever(MultiQueryDeduplicationMixin):
    """
    Retriever that runs multiple queries with hybrid search and deduplicates results.

    This retriever is more complex than Chroma/BM25 because it requires:
    1. Embedding generation (sparse + dense)
    2. Warmup for embedders
    3. Special handling for query preprocessing

    Uses MultiQueryDeduplicationMixin directly (not BaseMultiQueryRetriever)
    because it needs custom run() logic for embeddings.

    Example:
        retriever = MultiQueryQdrantHybridRetriever(
            document_store=qdrant_store,
            top_k=3
        )
        retriever.warm_up()  # Load embedders
        result = retriever.run(queries=["query1", "query2"])
    """

    def __init__(
        self,
        document_store: Any,
        top_k: int = 3,
        min_score: float | None = None,
        sparse_embedder_model: str = "Qdrant/bm42-all-minilm-l6-v2-attentions",
        dense_embedder_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ) -> None:
        self.document_store = document_store
        self.top_k = top_k
        self.min_score = min_score
        self.sparse_embedder_model = sparse_embedder_model
        self.dense_embedder_model = dense_embedder_model
        self._sparse_embedder: Any | None = None
        self._dense_embedder: Any | None = None

    def warm_up(self) -> None:
        """Initialize and warm up the embedding models."""

        self._sparse_embedder = FastembedSparseTextEmbedder(
            model=self.sparse_embedder_model,
        )
        self._sparse_embedder.warm_up()

        self._dense_embedder = FastembedTextEmbedder(
            model=self.dense_embedder_model,
        )
        self._dense_embedder.warm_up()

    @component.output_types(documents=list[HaystackDocument])
    def run(
        self, queries: str | list[str], top_k: int | None = None
    ) -> dict[str, list[HaystackDocument]]:
        """
        Run hybrid search with multiple queries.

        Args:
            queries: Single query string or list of query strings
            top_k: Maximum documents to return

        Returns:
            Dict with deduplicated documents sorted by score
        """
        if self._sparse_embedder is None:
            self.warm_up()

        if self._sparse_embedder is None or self._dense_embedder is None:
            raise ValueError("Embedders not initialized after warm_up()")

        # Convert single query to list
        if isinstance(queries, str):
            queries = [queries]

        k = top_k if top_k is not None else self.top_k

        # Run hybrid search for all queries
        results: list[dict[str, list[HaystackDocument]]] = []
        for query in queries:
            sparse_result = self._sparse_embedder.run(text=query)
            dense_result = self._dense_embedder.run(text=query)
            retriever = QdrantHybridRetriever(
                document_store=self.document_store,
                score_threshold=self.min_score,
            )
            result = retriever.run(
                query_sparse_embedding=sparse_result["sparse_embedding"],
                query_embedding=dense_result["embedding"],
                top_k=k,
            )
            results.append(result)

        # Deduplicate and rank using mixin
        docs = self.deduplicate_and_rank(results, k, self.min_score)
        return {"documents": docs}

    @component.output_types(documents=list[HaystackDocument])
    async def run_async(
        self, queries: str | list[str], top_k: int | None = None
    ) -> dict[str, list[HaystackDocument]]:
        if isinstance(queries, str):
            queries = [queries]
        k = top_k if top_k is not None else self.top_k
        if self._sparse_embedder is None:
            await asyncio.to_thread(self.warm_up)
        if self._sparse_embedder is None or self._dense_embedder is None:
            raise ValueError("Embedders not initialized after warm_up()")
        results: list[dict[str, list[HaystackDocument]]] = []
        for query in queries:
            sparse_result = await asyncio.to_thread(self._sparse_embedder.run, text=query)
            dense_result = await asyncio.to_thread(self._dense_embedder.run, text=query)
            retriever = QdrantHybridRetriever(
                document_store=self.document_store,
                score_threshold=self.min_score,
            )
            result = await asyncio.to_thread(
                retriever.run,
                query_sparse_embedding=sparse_result["sparse_embedding"],
                query_embedding=dense_result["embedding"],
                top_k=k,
            )
            results.append(result)
        docs = self.deduplicate_and_rank(results, k, self.min_score)
        return {"documents": docs}
