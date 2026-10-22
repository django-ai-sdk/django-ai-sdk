from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from django_ai_sdk.logger import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from haystack import Pipeline
    from haystack.components.query import QueryExpander
    from haystack.tools import ComponentTool

    from django_ai_sdk.rags.schemas import RagDocument, ToolSpec

# TODO: move into prompts.py file, this should make maintenance easier.
DEFAULT_EXPANDER_PROMPT = """
You are a search query expansion agent.

Your task is to generate search queries based on the user's original query to improve search recall.

The goal is to capture different ways users might phrase the same question, including the original query itself.

The documents being searched are about: {{context}}

RULES:
1. Output exactly {{n_expansions}} queries
2. The FIRST query MUST be the original query verbatim
3. The remaining queries should focus on different aspects or use different terminology
4. Use the SAME LANGUAGE as the original query
5. Make queries natural and conversational
6. Keep every query about what the documents are about: the same meaning in their
   terms, never another sense of a word (an acronym or name means what it would
   mean in these documents)

Original query: {{query}}

Respond with only a JSON object containing a "queries" array, no other text.
Example: {"queries": ["original query", "variation 1", "variation 2"]}
"""


#: What the expander is told about the searched documents (see `expander_context`).
CONTEXT_NAMES = 10
CONTEXT_KEYWORDS = 30


def expander_context(documents: list[RagDocument]) -> str:
    """What the searched documents are about, for the query expander: their most
    common keywords and some of their names. Without it, an expander guesses: "SABRE"
    in a set of CVs became the airline booking system."""
    keywords = Counter(
        keyword.strip()
        for document in documents
        for keyword in (document.metadata.get("keywords") or "").split(". ")
        if keyword.strip()
    )
    names = [n for d in documents if (n := d.metadata.get("name") or d.title)][:CONTEXT_NAMES]
    parts = []
    if keywords:
        parts.append(", ".join(k for k, _ in keywords.most_common(CONTEXT_KEYWORDS)))
    if names:
        parts.append(f"(files: {', '.join(names)})")
    return " ".join(parts) or "unknown"


def with_context(template: str, context: str) -> str:
    """`template` with `{{context}}` filled in now; the expander only fills in
    `{{query}}` and `{{n_expansions}}`. The context is kept out of Jinja."""
    safe = context.replace("{", "(").replace("}", ")")
    return template.replace("{{context}}", "{% raw %}" + safe + "{% endraw %}")


class RAGConfig(BaseModel):
    """
    Base configuration for Haystack RAG implementations.

    This provides common configuration options for all Haystack-based RAG
    implementations, including query expansion settings.

    Attributes:
        top_k: Maximum number of documents to retrieve per query
        n_expansions: Number of query variations to generate (1 = no expansion)
        expander_model: LLM model to use for query expansion
        expander_prompt: Prompt template for query expansion
        chunk_size: Chunk size for document splitting
        chunk_overlap: Chunk overlap for document splitting
    """

    top_k: int = Field(default=5, ge=1, description="Maximum documents to retrieve per query")
    min_score: float | None = Field(
        default=None,
        description="Drop documents below this relevance score. None disables filtering.",
    )
    n_expansions: int = Field(
        default=4,
        ge=1,
        description="Number of query variations to generate (1 = no expansion)",
    )
    expander_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model to use for query expansion",
    )
    expander_prompt: str = Field(
        default=DEFAULT_EXPANDER_PROMPT,
        description="Prompt template for query expansion; `{{context}}` gets what the "
        "searched documents are about",
    )
    chunk_size: int = Field(
        default=100,
        ge=1,
        description="Chunk size for document splitting",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        description="Chunk overlap for document splitting",
    )


class RAGBase[ConfigT: RAGConfig](ABC):
    """Abstract base class for Haystack RAG implementations."""

    _is_warmed_up: bool = False
    config: ConfigT
    documents: list[RagDocument]

    def get_query_expander(self) -> QueryExpander:
        """The query expander, told what this RAG's documents are about."""
        from haystack.components.query import QueryExpander  # noqa: PLC0415

        from django_ai_sdk.generators import openai_chat  # noqa: PLC0415

        return QueryExpander(
            chat_generator=openai_chat(model=self.config.expander_model),
            n_expansions=self.config.n_expansions,
            prompt_template=with_context(
                self.config.expander_prompt, expander_context(self.documents)
            ),
        )

    @abstractmethod
    async def warmup(self, force_rebuild: bool = False) -> None:
        """
        Warm up the RAG by building the indexed document store (expensive).

        After warmup, subsequent build_pipeline() calls will use the cached store.

        Args:
            force_rebuild: If True, clears existing index and rebuilds from scratch.
                          For persistent storage backends (like Qdrant), this will
                          delete and recreate the entire index.
        """
        pass

    @property
    def needs_warmup(self) -> bool:
        """Check if warmup is needed."""
        return not self._is_warmed_up

    @abstractmethod
    async def build_pipeline(self) -> Pipeline:
        """
        Build and return the RAG pipeline (query side, cheap).

        Returns:
            A Haystack Pipeline configured for RAG.
        """
        pass

    @abstractmethod
    async def as_tool(self) -> ComponentTool:
        """
        Return the RAG pipeline wrapped as a ComponentTool.

        Returns:
            A ComponentTool wrapping the RAG pipeline.
        """
        pass

    async def add_documents(self, documents: list[RagDocument]) -> None:
        """
        Add documents to an existing index (incremental update).

        Subclasses should override this if they support incremental adds.
        Default is a no-op (forces full reindex via warmup).

        Args:
            documents: List of RagDocument objects to add.
        """
        logger.warning(
            f"{self.__class__.__name__} does not support incremental add, use refresh_documents()"
        )

    async def remove_documents(self, document_ids: list[str]) -> None:
        """
        Remove documents from an existing index (incremental update).

        Subclasses should override this if they support incremental removal.
        Default is a no-op (forces full reindex via warmup).

        Args:
            document_ids: List of document IDs to remove.
        """
        logger.warning(
            f"{self.__class__.__name__} does not support incremental remove, use refresh_documents()"
        )

    async def refresh_documents(self, documents: list[RagDocument]) -> None:
        """
        Fully refresh the index with a new set of documents.

        Replaces all documents in the index. Subclasses should override
        this to perform an efficient in-place refresh without releasing
        file locks (for persistent backends).

        Default implementation calls warmup(force_rebuild=True).

        Args:
            documents: The new complete set of documents.
        """
        logger.debug(f"Refreshing documents for {self.__class__.__name__}")
        self.documents = documents
        await self.warmup(force_rebuild=True)

    async def get_chunk(self, chunk_id: str) -> str | None:
        """Return the content of a specific chunk by its Haystack document ID.
        Subclasses can override this if their store supports direct chunk lookup.
        Returns None to signal the caller should fall back to full Entry content.
        """
        return None

    async def get_tool(self, spec: ToolSpec) -> ComponentTool:
        """Get tool with custom specification."""
        tool = await self.as_tool()
        tool.name = spec.name
        tool.description = spec.description
        return tool
