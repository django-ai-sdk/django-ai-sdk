from __future__ import annotations

from typing import TYPE_CHECKING

from haystack import Pipeline
from haystack.components.query import QueryExpander
from haystack.components.writers import DocumentWriter
from haystack.core.super_component import SuperComponent
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.tools import ComponentTool

from django_ai_sdk.generators import openai_chat
from django_ai_sdk.logger import get_logger
from django_ai_sdk.rags.base import RAGBase, RAGConfig
from django_ai_sdk.rags.components import MultiQueryBM25Retriever
from django_ai_sdk.rags.utils import to_document

if TYPE_CHECKING:
    from haystack.dataclasses import Document as HaystackDocument

    from django_ai_sdk.rags.schemas import RagDocument

logger = get_logger(__name__)


class BM25QueryExpanderRAGConfig(RAGConfig):
    """Configuration for BM25 Query Expander RAG."""

    pass


class BM25QueryExpanderRAG(RAGBase):
    """RAG implementation using BM25 with optional query expansion."""

    def __init__(
        self,
        documents: list[RagDocument],
        config: BM25QueryExpanderRAGConfig | None = None,
    ) -> None:
        self.config = config or BM25QueryExpanderRAGConfig()
        self.documents = documents
        self._cached_document_store: InMemoryDocumentStore | None = None
        self._is_warmed_up = False
        logger.debug(f"BM25QueryExpanderRAG initialized with {len(documents)} documents")

    def _convert_documents(self) -> list[HaystackDocument]:
        """Convert RagDocuments to HaystackDocuments for internal use."""
        return [to_document(doc) for doc in self.documents]

    def _create_document_store(self) -> InMemoryDocumentStore:
        """Create an in-memory document store."""
        return InMemoryDocumentStore()

    def _has_existing_index(self, document_store: InMemoryDocumentStore) -> bool:
        """Check if document store already has indexed documents."""
        return document_store.count_documents() > 0

    async def add_documents(self, documents: list[RagDocument]) -> None:
        """Add documents to the existing BM25 index."""
        if self._cached_document_store is None:
            logger.warning("No document store available, cannot add documents")
            return

        haystack_docs = [to_document(doc) for doc in documents]
        self._write_documents(haystack_docs, self._cached_document_store)
        logger.info(f"Added {len(documents)} documents to BM25 index")

    async def remove_documents(self, document_ids: list[str]) -> None:
        """Remove documents from the BM25 index."""
        if self._cached_document_store is None:
            logger.warning("No document store available, cannot remove documents")
            return

        existing = self._cached_document_store.filter_documents()
        remaining = [d for d in existing if d.id not in document_ids]

        self._cached_document_store.delete_documents(document_ids=[d.id for d in existing])
        if remaining:
            self._write_documents(remaining, self._cached_document_store)

        logger.info(f"Removed {len(document_ids)} documents from BM25 index")

    def _write_documents(
        self, documents: list[HaystackDocument], document_store: InMemoryDocumentStore
    ) -> None:
        """Write documents to the store (no chunking for BM25)."""
        writer = DocumentWriter(document_store)
        writer.run(documents=documents)

    async def warmup(self, force_rebuild: bool = False) -> None:
        """
        Build or load indexed document store.

        Args:
            force_rebuild: If True, clears existing index and rebuilds from scratch.
        """
        if self._is_warmed_up and not force_rebuild:
            logger.debug("BM25QueryExpanderRAG already warmed up, skipping")
            return

        if force_rebuild:
            logger.info("Force rebuild requested, resetting BM25 index")
            self._is_warmed_up = False

        logger.debug("Warming up BM25QueryExpanderRAG - building indexed document store")
        logger.info(
            f"[warmup] force_rebuild={force_rebuild}, source_documents={len(self.documents)}"
        )

        document_store = self._create_document_store()

        if not force_rebuild and self._has_existing_index(document_store):
            existing_count = document_store.count_documents()
            self._cached_document_store = document_store
            self._is_warmed_up = True
            logger.info(f"Using existing BM25 index with {existing_count} documents")
            return

        haystack_docs = self._convert_documents()
        logger.info(f"[warmup] Converted {len(haystack_docs)} HaystackDocuments")

        logger.debug(f"Writing {len(haystack_docs)} documents to BM25 store")
        self._write_documents(haystack_docs, document_store)

        self._cached_document_store = document_store
        self._is_warmed_up = True
        indexed_count = document_store.count_documents()
        logger.info(
            f"BM25QueryExpanderRAG warmup complete: {len(self.documents)} source docs -> {indexed_count} docs indexed"
        )

    async def build_pipeline(self) -> Pipeline:
        """Build the RAG pipeline with BM25."""
        logger.debug("Building BM25 RAG query pipeline")

        if self._cached_document_store is not None:
            document_store = self._cached_document_store
        else:
            document_store = self._create_document_store()

            if not self._has_existing_index(document_store):
                haystack_docs = self._convert_documents()
                self._write_documents(haystack_docs, document_store)

        expander_generator = openai_chat(model=self.config.expander_model)

        query_expander = QueryExpander(
            chat_generator=expander_generator,
            n_expansions=self.config.n_expansions,
            prompt_template=self.config.expander_prompt,
        )

        retriever = MultiQueryBM25Retriever(
            document_store=document_store,
            top_k=self.config.top_k,
        )

        pipeline = Pipeline()
        pipeline.add_component("expander", query_expander)
        pipeline.add_component("retriever", retriever)

        pipeline.connect("expander.queries", "retriever.queries")
        logger.debug("BM25 RAG pipeline built successfully")
        return pipeline

    async def refresh_documents(self, documents: list[RagDocument]) -> None:
        """
        Update the indexed documents in-place.

        Replaces self.documents with the new list and re-indexes all of them
        into the already-open document store.
        """
        self.documents = documents
        logger.info(f"[refresh_documents] Refreshing BM25 index with {len(documents)} documents")

        if self._cached_document_store is None:
            await self.warmup(force_rebuild=True)
            return

        document_store = self._cached_document_store

        existing_docs = document_store.filter_documents()
        if existing_docs:
            document_store.delete_documents(document_ids=[doc.id for doc in existing_docs])

        haystack_docs = self._convert_documents()
        self._write_documents(haystack_docs, document_store)

        indexed_count = document_store.count_documents()
        logger.info(
            f"[refresh_documents] Done: {len(documents)} source docs -> {indexed_count} docs"
        )

    async def as_tool(self) -> ComponentTool:
        """Return the RAG pipeline as a ComponentTool."""
        if self.needs_warmup:
            logger.debug("RAG needs warmup before creating tool, warming up now")
            await self.warmup()

        logger.debug("Creating BM25 RAG pipeline as ComponentTool")
        pipeline = await self.build_pipeline()

        rag_super = SuperComponent(
            pipeline=pipeline,
            input_mapping={"query": ["expander.query"]},
            output_mapping={"retriever.documents": "documents"},
        )

        logger.debug("BM25 RAG ComponentTool created successfully")
        return ComponentTool(
            component=rag_super,
            name="bm25_rag_tool",
            description="Retrieves relevant documents using BM25 keyword search.",
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
