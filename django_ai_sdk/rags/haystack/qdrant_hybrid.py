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
from haystack_integrations.components.embedders.fastembed import (
    FastembedDocumentEmbedder,
    FastembedSparseDocumentEmbedder,
)
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from pydantic import Field

from django_ai_sdk.logger import get_logger
from django_ai_sdk.rags.haystack.base import BaseHaystackRAGConfig, HaystackRAGBase
from django_ai_sdk.rags.haystack.components import MultiQueryQdrantHybridRetriever
from django_ai_sdk.rags.schemas import RagDocument
from django_ai_sdk.rags.utils import rag_document_to_haystack

logger = get_logger(__name__)


class QdrantBM25HybridRAGConfig(BaseHaystackRAGConfig):
    """Configuration for Qdrant Hybrid RAG (BM42 Sparse + Dense)."""

    # Preselected multilingual models
    sparse_embedder_model: str = Field(
        default="Qdrant/bm42-all-minilm-l6-v2-attentions",
        description="Sparse embedder model for BM42 keyword matching",
    )
    dense_embedder_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="Dense embedder model for semantic matching",
    )

    # Qdrant's BM42-based settings
    embedding_dim: int = Field(default=384, ge=1, description="Embedding dimension")
    chunk_size: int = Field(default=500, ge=1, description="Document chunk size")
    chunk_overlap: int = Field(default=150, ge=0, description="Overlap between chunks")
    meta_fields_to_embed: list[str] = Field(
        default=["title"],
        description="Metadata fields to include in embeddings",
    )


class QdrantBM25HybridRAG(HaystackRAGBase):
    """
    RAG implementation using Qdrant with Hybrid retrieval + Query Expansion.

    Combines:
    - QueryExpander: generates multiple query variations
    - Sparse retrieval (BM42) for keyword-based matching (multilingual)
    - Dense retrieval (paraphrase-multilingual-MiniLM-L12-v2) for semantic matching
    - Reciprocal Rank Fusion (RRF) for combining results
    """

    def __init__(
        self,
        documents: list[RagDocument],
        config: QdrantBM25HybridRAGConfig | None = None,
    ) -> None:
        self.config = config or QdrantBM25HybridRAGConfig()
        self.documents = documents
        self._cached_document_store = None
        self._is_warmed_up = False
        logger.debug(f"QdrantBM25HybridRAG initialized with {len(documents)} documents")

    def _convert_documents(self) -> list[HaystackDocument]:
        """Convert RagDocuments to HaystackDocuments for internal use."""
        return [rag_document_to_haystack(doc) for doc in self.documents]

    def warmup(self) -> None:
        """Build and cache the indexed document store (expensive)."""
        if self._is_warmed_up:
            logger.debug("QdrantBM25HybridRAG already warmed up, skipping")
            return

        logger.debug("Warming up QdrantBM25HybridRAG - building indexed document store")

        document_store = QdrantDocumentStore(
            ":memory:",
            recreate_index=True,
            return_embedding=True,
            use_sparse_embeddings=True,
            embedding_dim=self.config.embedding_dim,
        )

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

        # Convert RagDocuments to HaystackDocuments
        haystack_docs = self._convert_documents()

        logger.debug(
            f"Writing {len(haystack_docs)} documents to Qdrant with chunking (size={self.config.chunk_size}, overlap={self.config.chunk_overlap})"
        )
        indexing_pipeline.run({"documents": haystack_docs})

        self._cached_document_store = document_store
        self._is_warmed_up = True
        logger.debug("QdrantBM25HybridRAG warmup complete")

    def build_pipeline(self) -> Pipeline:
        """Build the RAG pipeline with Qdrant Hybrid retrieval and query expansion."""
        logger.debug("Building Qdrant Hybrid RAG query pipeline")

        if self._cached_document_store is not None:
            document_store = self._cached_document_store
            logger.debug("Using cached document store")
        else:
            logger.debug("No cached document store, building fresh")
            document_store = QdrantDocumentStore(
                ":memory:",
                recreate_index=True,
                return_embedding=True,
                use_sparse_embeddings=True,
                embedding_dim=self.config.embedding_dim,
            )
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

            # Convert RagDocuments to HaystackDocuments
            haystack_docs = self._convert_documents()

            logger.debug(
                f"Writing {len(haystack_docs)} documents to Qdrant with chunking (size={self.config.chunk_size}, overlap={self.config.chunk_overlap})"
            )
            indexing_pipeline.run({"documents": haystack_docs})

        expander_generator = OpenAIChatGenerator(
            model=self.config.expander_model,
            # TODO: make this configurable
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            api_base_url=getattr(settings, "OPENAI_API_URL", None),
        )

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
                sparse_embedder_model=self.config.sparse_embedder_model,
                dense_embedder_model=self.config.dense_embedder_model,
            ),
        )

        query_pipeline.connect("expander.queries", "retriever.queries")

        logger.debug("Qdrant Hybrid RAG pipeline built successfully with query expansion")
        return query_pipeline

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
