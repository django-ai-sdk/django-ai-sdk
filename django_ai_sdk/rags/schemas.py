from typing import Any

from pydantic import BaseModel, Field


class RagDocument(BaseModel):
    """
    Framework-agnostic document for RAG implementations.

    Works with Haystack, LangChain, custom RAGs, or any other framework.
    Minimal required fields with flexible metadata support.

    Attributes:
        id: Unique document identifier (auto-generated UUID if not provided)
        content: Document text content
        metadata: Flexible metadata dictionary for framework-specific fields
        title: Optional document title
        source: Optional source reference (URL, file path, etc.)
        score: Optional relevance score (set during retrieval)

    Example:
        # Create from scratch
        doc = RagDocument(
            id="doc1",
            content="Python is a programming language...",
            metadata={"source": "docs.python.org", "topic": "programming"}
        )

        # Create from dict
        doc = RagDocument.from_dict({
            "id": "doc2",
            "content": "Django is a web framework...",
            "metadata": {"category": "web"}
        })
    """

    id: str = ""
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    title: str | None = None
    source: str | None = None
    score: float | None = None

    class Config:
        extra = "allow"

    # TODO: make vendor specific like `from_haystack`
    @classmethod
    def from_dict(cls, data: dict) -> "RagDocument":
        """
        Create RagDocument from a dictionary.
        Handles various dict formats from different sources:
        - Standard: {"id": "...", "content": "...", "metadata": {...}}
        - Haystack-style: {"id": "...", "content": "...", "meta": {...}}
        - LangChain-style: {"page_content": "...", "metadata": {...}}

        Args:
            data: Dictionary with document data

        Returns:
            RagDocument instance
        """
        # Handle different content field names
        content = data.get("content") or data.get("page_content", "")

        # Handle different metadata field names
        metadata = data.get("metadata") or data.get("meta", {})

        return cls(
            id=str(data.get("id", "")),
            content=content,
            metadata=metadata if isinstance(metadata, dict) else {},
            title=data.get("title"),
            source=data.get("source"),
        )

    @classmethod
    def from_haystack(cls, haystack_doc: Any) -> "RagDocument":
        """
        Create RagDocument from a Haystack Document.

        Args:
            haystack_doc: Haystack Document object

        Returns:
            RagDocument instance
        """
        meta = getattr(haystack_doc, "meta", {}) or {}

        return cls(
            id=str(haystack_doc.id),
            content=haystack_doc.content,
            metadata=meta,
            title=meta.get("title"),
            source=meta.get("source"),
        )

    # TODO: mark as obsolete, we should use pydantic dump methods.
    def to_dict(self) -> dict:
        """
        Convert to dictionary format.

        Returns:
            Dictionary representation of the document
        """
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "title": self.title,
            "source": self.source,
            "score": self.score,
        }


class ToolSpec(BaseModel):
    """Specification for RAG tool presentation to LLM."""

    name: str = Field(description="Tool name for function calling")
    description: str = Field(description="Tool description for LLM")
    doc_count: int | None = Field(default=None, description="Number of documents")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")
