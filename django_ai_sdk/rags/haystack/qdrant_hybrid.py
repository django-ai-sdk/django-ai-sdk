from django.conf import settings
from haystack import AsyncPipeline
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.preprocessors import RecursiveDocumentSplitter
from haystack.components.query import QueryExpander
from haystack.components.writers import DocumentWriter
from haystack.core.super_component import SuperComponent
from haystack.dataclasses import Document as HaystackDocument
from haystack.document_stores.types import DuplicatePolicy
from haystack.tools import ComponentTool
from haystack.utils import Secret
from haystack_integrations.components.embedders.fastembed import (
    FastembedDocumentEmbedder,
    FastembedSparseDocumentEmbedder,
)
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from pydantic import Field

from django_ai_sdk.logger import get_logger
from django_ai_sdk.rags.config import QdrantStorageConfig
from django_ai_sdk.rags.haystack.base import BaseHaystackRAGConfig, HaystackRAGBase
from django_ai_sdk.rags.haystack.components import MultiQueryQdrantHybridRetriever
from django_ai_sdk.rags.schemas import RagDocument
from django_ai_sdk.rags.utils import rag_document_to_haystack

logger = get_logger(__name__)


class QdrantBM25HybridRAGConfig(BaseHaystackRAGConfig):
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


class QdrantBM25HybridRAG(HaystackRAGBase):
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
        return [rag_document_to_haystack(doc) for doc in self.documents]

    def _create_document_store(self, recreate: bool = False) -> QdrantDocumentStore:
        """Create document store based on persistence configuration."""
        import os

        storage = self.config.storage

        if storage.is_persistent and storage.persist_path:
            os.makedirs(storage.persist_path, exist_ok=True)

            return QdrantDocumentStore(
                path=storage.persist_path,
                index="documents",
                recreate_index=recreate,
                return_embedding=True,
                use_sparse_embeddings=True,
                embedding_dim=self.config.embedding_dim,
                on_disk=storage.qdrant_on_disk,
                similarity=storage.qdrant_similarity,
            )
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
        return document_store.count_documents() > 0

    async def add_documents(self, documents: list["RagDocument"]) -> None:
        """Add documents to the existing Qdrant index."""
        if self._cached_document_store is None:
            logger.warning("No document store available, cannot add documents")
            return

        haystack_docs = [rag_document_to_haystack(doc) for doc in documents]
        self._index_documents(haystack_docs, self._cached_document_store)
        logger.info(f"Added {len(documents)} documents to Qdrant index")

    def _index_documents(self, documents: list, document_store: QdrantDocumentStore) -> None:
        """Index documents with chunking and embedding."""

        # Add original doc_id to metadata so we can delete by it later
        for doc in documents:
            if "doc_id" not in doc.meta:
                doc.meta["doc_id"] = doc.id

        indexing_pipeline = AsyncPipeline()
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

        indexing_pipeline.run({"documents": documents})

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

            self._cached_document_store.delete_by_filter(filters=filter_obj)
            logger.info(f"Removed {len(document_ids)} documents from Qdrant index")
        except Exception as e:
            logger.error(f"Failed to remove documents: {e}")

    def warmup(self, force_rebuild: bool = False) -> None:
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

        if not force_rebuild and storage.is_persistent and self._has_existing_index(document_store):
            existing_count = document_store.count_documents()
            self._cached_document_store = document_store
            self._is_warmed_up = True
            logger.info(
                f"Using existing Qdrant index from {storage.persist_path} with {existing_count} chunks"
            )
            return

        logger.info(
            f"Creating new Qdrant index for persistent storage at {storage.persist_path}"
            if storage.is_persistent
            else "Creating in-memory Qdrant index"
        )

        # Convert RagDocuments to HaystackDocuments
        haystack_docs = self._convert_documents()
        logger.info(f"[warmup] Converted {len(haystack_docs)} HaystackDocuments")

        logger.debug(
            f"Writing {len(haystack_docs)} documents to Qdrant with chunking (size={self.config.chunk_size}, overlap={self.config.chunk_overlap})"
        )
        self._index_documents(haystack_docs, document_store)

        indexed_count = document_store.count_documents()
        self._cached_document_store = document_store
        self._is_warmed_up = True
        logger.info(
            f"QdrantBM25HybridRAG warmup complete: {len(self.documents)} source docs → {indexed_count} chunks indexed"
        )

    def build_pipeline(self) -> AsyncPipeline:
        logger.debug("Building Qdrant Hybrid RAG query pipeline")

        if self._cached_document_store is not None:
            document_store = self._cached_document_store
        else:
            document_store = self._create_document_store(recreate=False)

            if not self._has_existing_index(document_store):
                haystack_docs = self._convert_documents()
                self._index_documents(haystack_docs, document_store)

        expander_generator = OpenAIChatGenerator(
            model=self.config.expander_model,
            api_key=Secret.from_token(settings.OPENAI_API_KEY),
            api_base_url=getattr(settings, "OPENAI_API_URL", None),
        )

        query_expander = QueryExpander(
            chat_generator=expander_generator,
            n_expansions=self.config.n_expansions,
            prompt_template=self.config.expander_prompt,
        )

        query_pipeline = AsyncPipeline()
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

    def refresh_documents(self, documents: list[RagDocument]) -> None:
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
            self.warmup(force_rebuild=True)
            return

        document_store = self._cached_document_store

        # Wipe and rewrite all chunks so deleted/updated docs don't linger
        existing_docs = document_store.filter_documents()
        if existing_docs:
            document_store.delete_documents(document_ids=[doc.id for doc in existing_docs])

        # Convert and index using shared helper
        haystack_docs = self._convert_documents()
        self._index_documents(haystack_docs, document_store)

        indexed_count = document_store.count_documents()
        logger.info(
            f"[refresh_documents] Done: {len(documents)} source docs → {indexed_count} chunks"
        )

    def as_tool(self) -> ComponentTool:
        """Return the RAG pipeline as a ComponentTool."""
        if self.needs_warmup:
            logger.debug("RAG needs warmup before creating tool, warming up now")
            self.warmup()

        logger.debug("Creating Qdrant Hybrid RAG pipeline as ComponentTool")
        pipeline = self.build_pipeline()

        rag_super = SuperComponent(
            pipeline=pipeline,
            input_mapping={"query": ["expander.query"]},
            output_mapping={"retriever.documents": "documents"},
        )

        logger.debug("Qdrant Hybrid RAG ComponentTool created successfully")

        tool = ComponentTool(
            component=rag_super,
            name="hybrid_rag_tool",
            description="Retrieves relevant documents using hybrid search with query expansion.",
        )

        return tool

    async def get_chunk(self, chunk_id: str) -> str | None:
        """Fetch a single chunk from the Qdrant store by its Haystack document ID.

        Only uses the already-open cached store — never opens a second connection,
        which would conflict with the exclusive file lock held during RAG warmup.
        """
        if self._cached_document_store is None:
            return None
        docs = self._cached_document_store.get_documents_by_id([chunk_id])
        return docs[0].content if docs else None
