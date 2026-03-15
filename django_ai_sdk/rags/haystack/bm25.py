"""
BM25-based RAG implementation with query expansion.
"""

from django.conf import settings
from haystack import Pipeline
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.query import QueryExpander
from haystack.components.writers import DocumentWriter
from haystack.core.super_component import SuperComponent
from haystack.dataclasses import Document as HaystackDocument
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.tools import ComponentTool
from haystack.utils import Secret
from pydantic import BaseModel

from django_ai_sdk.logger import get_logger
from django_ai_sdk.rags.haystack.base import HaystackRAGBase
from django_ai_sdk.rags.schemas import RagDocument
from django_ai_sdk.rags.utils import rag_document_to_haystack

logger = get_logger(__name__)


class BM25QueryExpanderRAGConfig(BaseModel):
    """Configuration for BM25 Query Expander RAG."""

    top_k: int = 5
    n_expansions: int = 4


class BM25QueryExpanderRAG(HaystackRAGBase):
    """RAG implementation using BM25 with optional query expansion."""

    def __init__(
        self,
        documents: list[RagDocument],
        config: BM25QueryExpanderRAGConfig | None = None,
    ) -> None:
        self.config = config or BM25QueryExpanderRAGConfig()
        self.documents = documents
        self._cached_document_store = None
        self._is_warmed_up = False
        logger.debug(f"BM25QueryExpanderRAG initialized with {len(documents)} documents")

    def _convert_documents(self) -> list[HaystackDocument]:
        """Convert RagDocuments to HaystackDocuments for internal use."""
        return [rag_document_to_haystack(doc) for doc in self.documents]

    def warmup(self) -> None:
        """Build and cache the indexed document store (expensive)."""
        if self._is_warmed_up:
            logger.debug("BM25QueryExpanderRAG already warmed up, skipping")
            return

        logger.debug("Warming up BM25QueryExpanderRAG - building indexed document store")

        # DK: move to file stored version when procided path in config
        document_store = InMemoryDocumentStore()

        # Convert RagDocuments to HaystackDocuments
        haystack_docs = self._convert_documents()

        indexing_pipeline = Pipeline()
        indexing_pipeline.add_component("writer", DocumentWriter(document_store))
        logger.debug(f"Writing {len(haystack_docs)} documents to store")
        indexing_pipeline.run({"writer": {"documents": haystack_docs}})

        self._cached_document_store = document_store
        self._is_warmed_up = True
        logger.debug("BM25QueryExpanderRAG warmup complete")

    def build_pipeline(self) -> Pipeline:
        """Build the RAG pipeline with BM25."""
        logger.debug("Building BM25 RAG query pipeline")

        if self._cached_document_store is not None:
            document_store = self._cached_document_store
            logger.debug("Using cached document store")
        else:
            logger.debug("No cached document store, building fresh (warmup needed)")
            document_store = InMemoryDocumentStore()

            # Convert RagDocuments to HaystackDocuments
            haystack_docs = self._convert_documents()

            indexing_pipeline = Pipeline()
            indexing_pipeline.add_component("writer", DocumentWriter(document_store))
            logger.debug(f"Writing {len(haystack_docs)} documents to store")
            indexing_pipeline.run({"writer": {"documents": haystack_docs}})

        # Create query expander
        expander_generator = OpenAIChatGenerator(
            model="gpt-4o-mini",
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            api_base_url=getattr(settings, "OPENAI_API_URL", None),
        )

        # Custom prompt that forces same-language expansion
        expander_prompt = """You are a query expansion assistant. Generate {{n_expansions}} alternative search queries for the given user query.

IMPORTANT: 
- Generate queries ONLY in the SAME language as the original query
- If the original query is in Dutch, generate ONLY Dutch queries
- If the original query is in English, generate ONLY English queries
- Do NOT mix languages
- Do NOT translate the queries

Return a JSON object with the key "queries" containing the list of queries.

Original query: {{query}}

Generate {{n_expansions}} alternative queries in the SAME language as the original:"""

        query_expander = QueryExpander(
            chat_generator=expander_generator,
            n_expansions=self.config.n_expansions,
            prompt_template=expander_prompt,
        )

        # Create BM25 retriever
        from django_ai_sdk.rags.haystack.components import MultiQueryBM25Retriever

        retriever = MultiQueryBM25Retriever(
            document_store=document_store,
            top_k=self.config.top_k,
        )

        # Build pipeline with query expander
        pipeline = Pipeline()
        pipeline.add_component("expander", query_expander)
        pipeline.add_component("retriever", retriever)

        # Connect: expander -> retriever
        pipeline.connect("expander.queries", "retriever.queries")

        logger.debug("BM25 RAG pipeline built successfully")
        return pipeline

    def as_tool(self) -> ComponentTool:
        """Return the RAG pipeline as a ComponentTool."""
        if self.needs_warmup:
            logger.debug("RAG needs warmup before creating tool, warming up now")
            self.warmup()

        logger.debug("Creating BM25 RAG pipeline as ComponentTool")
        pipeline = self.build_pipeline()

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
        )
