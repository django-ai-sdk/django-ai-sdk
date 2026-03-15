"""
Base RAG (Retrieval Augmented Generation) adapter for the Django AI SDK.

Provides a generic interface for RAG implementations that can retrieve
documents and format them for injection into LLM prompts.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class RAGSource(BaseModel):
    """Generic source document for citations."""

    id: str
    content: str
    metadata: dict[str, Any] = {}


class RAGResult(BaseModel):
    """Result from RAG retrieval."""

    documents: list[dict[str, Any]]
    context: str
    sources: list[RAGSource]
    query: str


class RAGConfig(BaseModel):
    """Configuration for RAG adapter."""

    embedder_model: str = "intfloat/multilingual-e5-large-instruct"
    top_k: int = 3
    document_threshold: float = 0.7


class BaseRAGAdapter(ABC):
    """
    Abstract base class for all RAG implementations (non-Haystack).

    Provides a generic interface for retrieving documents and formatting
    context for LLM injection. All custom/direct RAG implementations
    should inherit from this class.

    Key methods:
    - warmup(): Build search index (expensive, called once)
    - retrieve(): Search documents (fast, called per query)
    - format_context(): Format results for LLM

    Example:
        class MyRAG(BaseRAGAdapter):
            def warmup(self):
                self._index = build_index(self.documents)
                self._is_warmed_up = True

            async def retrieve(self, query: str) -> RAGResult:
                if self.needs_warmup:
                    self.warmup()
                results = self._index.search(query)
                return RAGResult(...)
    """

    _is_warmed_up: bool = False

    def __init__(self, config: BaseModel | None = None) -> None:
        """
        Initialize the RAG adapter.

        Args:
            config: Configuration model (RAGConfig, BM25Config, or custom)
        """
        self.config = config

    @abstractmethod
    def warmup(self) -> None:
        """
        Warm up the RAG by building the indexed document store (expensive).

        This should be called before the first retrieval, or after
        documents are modified. The base implementation is a no-op,
        subclasses should override with actual indexing logic.

        Example:
            if rag.needs_warmup:
                rag.warmup()  # Expensive, do once
        """
        pass

    @property
    def needs_warmup(self) -> bool:
        """Check if warmup is needed."""
        return not self._is_warmed_up

    @abstractmethod
    async def retrieve(self, query: str) -> RAGResult:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The search query string.

        Returns:
            RAGResult with documents, formatted context, and sources.
        """

    def format_context(self, result: RAGResult) -> str:
        """
        Format RAG result as context string for LLM injection.

        Args:
            result: The RAG result to format.

        Returns:
            Formatted context string suitable for system message.
        """
        if not result.documents:
            return ""

        context_parts = ["Context from retrieved documents:\n"]
        for i, doc in enumerate(result.documents, 1):
            content = doc.get("content", "")
            context_parts.append(f"\n--- Document {i} ---\n{content}")

        return "\n".join(context_parts)

    def get_retriever(
        self,
    ) -> Any:
        """
        Return a Haystack retriever component for pipeline integration.

        Override in subclasses that need Haystack compatibility.

        Returns:
            Haystack retriever component or None
        """
        return None
