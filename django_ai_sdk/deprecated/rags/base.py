from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from django_ai_sdk.logger import get_logger
from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Callable

    from django_ai_sdk.rags.schemas import RagDocument, ToolSpec

logger = get_logger(__name__)


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
    - as_tool(): Return tool for function calling
    - get_tool(spec): Return tool with custom specification

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

    @abstractmethod
    def as_tool(self) -> Callable:
        """
        Return the RAG as a tool callable for function calling.

        Returns:
            Callable that accepts query and returns documents.
        """
        pass

    def get_tool(self, spec: ToolSpec) -> Callable:
        """
        Get tool with custom specification.

        Args:
            spec: ToolSpec with name and description.

        Returns:
            Tool callable with customized name/description.
        """
        tool = self.as_tool()
        tool.name = spec.name
        tool.description = spec.description
        return tool

    async def add_documents(self, documents: list[RagDocument]) -> None:
        """
        Add documents incrementally (optional, override in subclass).

        Default falls back to full warmup.

        Args:
            documents: List of RagDocument objects to add.
        """
        logger.warning(f"{self.__class__.__name__} does not support incremental add, rebuilding")
        self.documents.extend(documents)
        if hasattr(self, "warmup"):
            self.warmup()

    async def remove_documents(self, document_ids: list[str]) -> None:
        """
        Remove documents incrementally (optional, override in subclass).

        Default falls back to filtering documents and full warmup.

        Args:
            document_ids: List of document IDs to remove.
        """
        logger.warning(f"{self.__class__.__name__} does not support incremental remove, rebuilding")
        removed = set(document_ids)
        self.documents = [d for d in self.documents if d.id not in removed]
        if hasattr(self, "warmup"):
            self.warmup()

    def refresh_documents(self, documents: list[RagDocument]) -> None:
        """
        Fully refresh the index with a new set of documents.

        Default calls warmup() which rebuilds the index from self.documents.

        Args:
            documents: The new complete set of documents.
        """
        self.documents = documents
        if hasattr(self, "warmup"):
            self.warmup()
