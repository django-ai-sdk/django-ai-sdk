from django.conf import settings
from haystack import Pipeline
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.preprocessors import RecursiveDocumentSplitter
from haystack.components.query import QueryExpander
from haystack.components.writers import DocumentWriter
from haystack.core.super_component import SuperComponent
from haystack.dataclasses import Document as HaystackDocument
from haystack.document_stores.types import DuplicatePolicy
from haystack.tools import ComponentTool
from haystack.utils import Secret
from haystack_integrations.components.embedders.fastembed import FastembedDocumentEmbedder
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from pydantic import Field

from django_ai_sdk.logger import get_logger
from django_ai_sdk.rags.config import ChromaStorageConfig
from django_ai_sdk.rags.haystack.base import BaseHaystackRAGConfig, HaystackRAGBase
from django_ai_sdk.rags.haystack.components import MultiQueryChromaRetriever
from django_ai_sdk.rags.schemas import RagDocument
from django_ai_sdk.rags.utils import rag_document_to_haystack

logger = get_logger(__name__)


class ChromaDBQueryExpanderRAGConfig(BaseHaystackRAGConfig):
    """Configuration for ChromaDB Query Expander RAG."""

    embedder_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model for document/query embeddings",
    )
    chunk_size: int = Field(default=260, ge=1, description="Size of document chunks")
    chunk_overlap: int = Field(default=0, ge=0, description="Overlap between consecutive chunks")
    meta_fields_to_embed: list[str] = Field(default=["title"])
    storage: ChromaStorageConfig = Field(default_factory=ChromaStorageConfig)


class ChromaDBQueryExpanderRAG(HaystackRAGBase):
    """RAG implementation using ChromaDB with query expansion."""

    def __init__(
        self,
        documents: list[RagDocument],
        config: ChromaDBQueryExpanderRAGConfig | None = None,
    ) -> None:
        self.config: ChromaDBQueryExpanderRAGConfig = config or ChromaDBQueryExpanderRAGConfig()
        self.documents = documents
        self._cached_document_store = None
        self._is_warmed_up = False
        logger.debug(f"ChromaDBQueryExpanderRAG initialized with {len(documents)} documents")

    def _convert_documents(self) -> list[HaystackDocument]:
        """Convert RagDocuments to HaystackDocuments for internal use."""
        return [rag_document_to_haystack(doc) for doc in self.documents]

    def _create_document_store(self, recreate: bool = False) -> ChromaDocumentStore:
        """Create document store based on persistence configuration."""
        import os
        import shutil

        if self.config.storage.is_persistent and self.config.storage.persist_path:
            if recreate and os.path.exists(self.config.storage.persist_path):
                shutil.rmtree(self.config.storage.persist_path)
                logger.info(f"Deleted existing Chroma index at {self.config.storage.persist_path}")
            os.makedirs(self.config.storage.persist_path, exist_ok=True)
            return ChromaDocumentStore(
                persist_path=self.config.storage.persist_path,
                distance=self.config.storage.chroma_distance,
            )
        else:
            return ChromaDocumentStore()

    def _has_existing_index(self, document_store: ChromaDocumentStore) -> bool:
        """Check if document store already has indexed documents."""
        return document_store.count_documents() > 0

    async def add_documents(self, documents: list[RagDocument]) -> None:
        """Add documents to the existing Chroma index."""
        if self._cached_document_store is None:
            logger.warning("No document store available, cannot add documents")
            return

        haystack_docs = [rag_document_to_haystack(doc) for doc in documents]
        self._index_documents(haystack_docs, self._cached_document_store)
        logger.info(f"Added {len(documents)} documents to Chroma index")

    def _index_documents(self, documents: list, document_store: ChromaDocumentStore) -> None:
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
            "doc_embedder",
            FastembedDocumentEmbedder(
                model=self.config.embedder_model,
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

        indexing_pipeline.connect("splitter", "doc_embedder")
        indexing_pipeline.connect("doc_embedder", "writer")

        indexing_pipeline.run({"documents": documents})

    async def remove_documents(self, document_ids: list[str]) -> None:
        """Remove documents from the Chroma index."""
        if self._cached_document_store is None:
            logger.warning("No document store available, cannot remove documents")
            return

        try:
            # Use filter_documents to find documents, then delete by IDs
            # Haystack filter format for Chroma
            if len(document_ids) == 1:
                filters = {"field": "meta.doc_id", "operator": "==", "value": document_ids[0]}
            else:
                filters = {"field": "meta.doc_id", "operator": "in", "value": document_ids}

            # Get matching documents
            matching_docs = self._cached_document_store.filter_documents(filters=filters)
            doc_ids_to_delete = [doc.id for doc in matching_docs]

            if doc_ids_to_delete:
                self._cached_document_store.delete_documents(document_ids=doc_ids_to_delete)

            logger.info(f"Removed {len(document_ids)} documents from Chroma index")
        except Exception as e:
            logger.error(f"Failed to remove documents: {e}")

    def warmup(self, force_rebuild: bool = False) -> None:
        """
        Build or load indexed document store.

        Args:
            force_rebuild: If True, clears existing index and rebuilds from scratch.
        """
        if self._is_warmed_up and not force_rebuild:
            logger.debug("ChromaDBQueryExpanderRAG already warmed up, skipping")
            return

        if force_rebuild:
            logger.info("Force rebuild requested, resetting Chroma index")
            self._is_warmed_up = False

        logger.debug("Warming up ChromaDBQueryExpanderRAG - building indexed document store")
        logger.info(
            f"[warmup] force_rebuild={force_rebuild}, source_documents={len(self.documents)}"
        )

        document_store = self._create_document_store(recreate=force_rebuild)

        if (
            not force_rebuild
            and self.config.storage.is_persistent
            and self._has_existing_index(document_store)
        ):
            existing_count = document_store.count_documents()
            self._cached_document_store = document_store
            self._is_warmed_up = True
            logger.info(
                f"Using existing Chroma index from {self.config.storage.persist_path} with {existing_count} chunks"
            )
            return

        logger.info(
            f"Creating new Chroma index for persistent storage at {self.config.storage.persist_path}"
            if self.config.storage.is_persistent
            else "Creating in-memory Chroma index"
        )

        haystack_docs = self._convert_documents()
        logger.info(f"[warmup] Converted {len(haystack_docs)} HaystackDocuments")

        logger.debug(
            f"Writing {len(haystack_docs)} documents to Chroma with chunking (size={self.config.chunk_size}, overlap={self.config.chunk_overlap})"
        )
        self._index_documents(haystack_docs, document_store)

        indexed_count = document_store.count_documents()
        self._cached_document_store = document_store
        self._is_warmed_up = True
        logger.info(
            f"ChromaDBQueryExpanderRAG warmup complete: {len(self.documents)} source docs -> {indexed_count} chunks indexed"
        )

    def build_pipeline(self) -> Pipeline:
        """Build the RAG pipeline with ChromaDB and query expansion."""
        logger.debug("Building ChromaDB RAG query pipeline")

        if self._cached_document_store is not None:
            document_store = self._cached_document_store
        else:
            document_store = self._create_document_store(recreate=False)

            if not self._has_existing_index(document_store):
                haystack_docs = self._convert_documents()
                self._index_documents(haystack_docs, document_store)

        expander_generator = OpenAIChatGenerator(
            model=self.config.expander_model,
            # TODO: we need to fix this
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            api_base_url=getattr(settings, "OPENAI_API_URL", None),
        )

        query_expander = QueryExpander(
            chat_generator=expander_generator,
            n_expansions=self.config.n_expansions,
            prompt_template=self.config.expander_prompt,
        )

        retriever = MultiQueryChromaRetriever(
            document_store=document_store,
            top_k=self.config.top_k,
        )

        pipeline = Pipeline()
        pipeline.add_component("expander", query_expander)
        pipeline.add_component("retriever", retriever)

        pipeline.connect("expander.queries", "retriever.queries")

        logger.debug("ChromaDB RAG pipeline built (retrieves documents only)")
        return pipeline

    def refresh_documents(self, documents: list[RagDocument]) -> None:
        """
        Update the indexed documents without releasing the Chroma file lock.

        Replaces self.documents with the new list and re-indexes all of them
        into the already-open document store using OVERWRITE policy.
        Safe to call on a warmed-up instance.
        """
        self.documents = documents
        logger.info(f"[refresh_documents] Refreshing Chroma index with {len(documents)} documents")

        if self._cached_document_store is None:
            self.warmup(force_rebuild=True)
            return

        document_store = self._cached_document_store

        existing_docs = document_store.filter_documents()
        if existing_docs:
            document_store.delete_documents(document_ids=[doc.id for doc in existing_docs])

        haystack_docs = self._convert_documents()
        self._index_documents(haystack_docs, document_store)

        indexed_count = document_store.count_documents()
        logger.info(
            f"[refresh_documents] Done: {len(documents)} source docs -> {indexed_count} chunks"
        )

    def as_tool(self) -> ComponentTool:
        """Return the RAG pipeline as a ComponentTool."""
        if self.needs_warmup:
            logger.debug("RAG needs warmup before creating tool, warming up now")
            self.warmup()

        logger.debug("Creating RAG pipeline as ComponentTool")
        pipeline = self.build_pipeline()

        rag_super = SuperComponent(
            pipeline=pipeline,
            input_mapping={"query": ["expander.query"]},
            output_mapping={"retriever.documents": "documents"},
        )

        logger.debug("RAG ComponentTool created successfully")

        return ComponentTool(
            component=rag_super,
            name="rag_tool",
            description="Retrieves relevant documents for answering questions.",
        )
