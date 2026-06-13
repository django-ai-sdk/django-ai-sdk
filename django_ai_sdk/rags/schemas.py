from typing import Any

from pydantic import BaseModel, Field


class RagDocument(BaseModel):
    """Framework-agnostic document for Haystack RAG implementations."""

    id: str = ""
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    title: str | None = None
    source: str | None = None
    score: float | None = None

    class Config:
        extra = "allow"

    @classmethod
    def from_document(cls, haystack_doc: Any) -> "RagDocument":
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

class ToolSpec(BaseModel):
    """Specification for RAG tool presentation to LLM."""

    name: str = Field(description="Tool name for function calling")
    description: str = Field(description="Tool description for LLM")
    doc_count: int | None = Field(default=None, description="Number of documents")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")
