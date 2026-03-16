from django.conf import settings
from haystack import Pipeline
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.preprocessors import RecursiveDocumentSplitter
from haystack.components.query import QueryExpander
from haystack.components.writers import DocumentWriter
from haystack.core.super_component import SuperComponent
from haystack.dataclasses import Document as HaystackDocument
from haystack.tools import ComponentTool
from haystack.utils import Secret
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from pydantic import BaseModel

from django_ai_sdk.logger import get_logger
from django_ai_sdk.rags.haystack.base import HaystackRAGBase
from django_ai_sdk.rags.schemas import RagDocument
from django_ai_sdk.rags.utils import rag_document_to_haystack

logger = get_logger(__name__)


DEFAULT_EXPANDER_PROMPT = """
You are a query expansion assistant. Generate {{n_expansions}} alternative search queries for the given user query.

IMPORTANT:
- Generate queries ONLY in the SAME language as the original query
- If the original query is in Dutch, generate ONLY Dutch queries
- If the original query is in English, generate ONLY English queries
- Do NOT mix languages
- Do NOT translate the queries

Return a JSON object with the key "queries" containing the list of queries.

Original query: {{query}}

Generate {{n_expansions}} alternative queries in the SAME language as the original:
"""


class ChromaDBQueryExpanderRAGConfig(BaseModel):
    """Configuration for ChromaDB Query Expander RAG."""

    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    expander_model: str = "gpt-4o-mini"
    expander_prompt: str = DEFAULT_EXPANDER_PROMPT
    top_k: int = 5
    n_expansions: int = 4
    chunk_size: int = 260
    chunk_overlap: int = 0


class ChromaDBQueryExpanderRAG(HaystackRAGBase):
    """RAG implementation using ChromaDB with query expansion."""

    def __init__(
        self,
        documents: list[RagDocument],
        config: ChromaDBQueryExpanderRAGConfig | None = None,
    ) -> None:
        self.config = config or ChromaDBQueryExpanderRAGConfig()
        self.documents = documents
        self._cached_document_store = None
        self._is_warmed_up = False
        logger.debug(f"ChromaDBQueryExpanderRAG initialized with {len(documents)} documents")

    def _convert_documents(self) -> list[HaystackDocument]:
        """Convert RagDocuments to HaystackDocuments for internal use."""
        return [rag_document_to_haystack(doc) for doc in self.documents]

    def warmup(self) -> None:
        """Build and cache the indexed document store (expensive)."""
        if self._is_warmed_up:
            logger.debug("ChromaDBQueryExpanderRAG already warmed up, skipping")
            return

        logger.debug("Warming up ChromaDBQueryExpanderRAG - building indexed document store")

        document_store = ChromaDocumentStore()

        indexing_pipeline = Pipeline()
        indexing_pipeline.add_component(
            "chunker",
            RecursiveDocumentSplitter(
                split_length=self.config.chunk_size,
                split_overlap=self.config.chunk_overlap,
                separators=["\n\n", "\n", ".", " "],
            ),
        )
        indexing_pipeline.add_component("writer", DocumentWriter(document_store))

        indexing_pipeline.connect("chunker", "writer")

        # Convert RagDocuments to HaystackDocuments
        haystack_docs = self._convert_documents()

        logger.debug(
            f"Writing {len(haystack_docs)} documents to ChromaDB with chunking (size={self.config.chunk_size}, overlap={self.config.chunk_overlap})"
        )
        indexing_pipeline.run({"chunker": {"documents": haystack_docs}})

        self._cached_document_store = document_store
        self._is_warmed_up = True
        logger.debug("ChromaDBQueryExpanderRAG warmup complete")

    def build_pipeline(self) -> Pipeline:
        """Build the RAG pipeline with ChromaDB and query expansion."""
        logger.debug("Building ChromaDB RAG query pipeline")

        if self._cached_document_store is not None:
            document_store = self._cached_document_store
            logger.debug("Using cached document store")
        else:
            logger.debug("No cached document store, building fresh (warmup needed)")
            document_store = ChromaDocumentStore()

            indexing_pipeline = Pipeline()
            indexing_pipeline.add_component(
                "chunker",
                RecursiveDocumentSplitter(
                    split_length=self.config.chunk_size,
                    split_overlap=self.config.chunk_overlap,
                    separators=["\n\n", "\n", ".", " "],
                ),
            )
            indexing_pipeline.add_component("writer", DocumentWriter(document_store))

            indexing_pipeline.connect("chunker", "writer")

            # Convert RagDocuments to HaystackDocuments
            haystack_docs = self._convert_documents()

            logger.debug(
                f"Writing {len(haystack_docs)} documents to ChromaDB with chunking (size={self.config.chunk_size}, overlap={self.config.chunk_overlap})"
            )
            indexing_pipeline.run({"chunker": {"documents": haystack_docs}})

        expander_generator = OpenAIChatGenerator(
            model=self.config.expander_model,
            # TODO: we need to fix this
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            api_base_url=getattr(settings, "OPENAI_API_URL", None),
        )

        # Create query expander with custom prompt to keep same language
        from django_ai_sdk.rags.haystack.components import MultiQueryChromaRetriever

        # Custom prompt that forces same-language expansion
        query_expander = QueryExpander(
            chat_generator=expander_generator,
            n_expansions=self.config.n_expansions,
            prompt_template=self.config.expander_prompt,
        )

        # Create retriever with query expansion support
        retriever = MultiQueryChromaRetriever(
            document_store=document_store,
            top_k=self.config.top_k,
        )

        # Build pipeline - RAG only retrieves documents, Agent handles generation
        pipeline = Pipeline()
        pipeline.add_component("expander", query_expander)
        pipeline.add_component("retriever", retriever)

        # Connect: expander -> retriever (returns documents)
        pipeline.connect("expander.queries", "retriever.queries")

        logger.debug("ChromaDB RAG pipeline built (retrieves documents only)")
        return pipeline

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
