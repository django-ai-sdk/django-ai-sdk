from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


class NumberedSource(BaseModel):
    """A source found during a turn"""

    index: int = 0
    key: str = ""
    title: str
    content: str
    kind: Literal["document", "web"] = "document"
    url: str | None = None
    chunk_id: str | None = None
    doc_id: str | None = None
    memory_id: str | None = None
    page_number: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def from_document(doc: Any) -> NumberedSource:
    """A source from a retrieved Haystack Document"""
    if hasattr(doc, "meta"):
        found: dict[str, Any] = {
            "chunk_id": doc.id,
            "content": doc.content or "",
            "meta": dict(doc.meta or {}),
        }
    else:
        found = dict(doc)

    meta: dict[str, Any] = dict(found.get("meta") or {})
    base = (
        meta.get("file_name")
        or meta.get("filename")
        or meta.get("name")
        or meta.get("title")
        or meta.get("topic")
        or "Document"
    )

    split_id = meta.get("split_id")
    return NumberedSource(
        title=f"{base} · §{split_id + 1}" if split_id is not None else base,
        content=str(found.get("content") or ""),
        chunk_id=found.get("chunk_id"),
        doc_id=meta.get("doc_id"),
        memory_id=meta.get("memory_id"),
        page_number=meta.get("page_number"),
        metadata={k: v for k, v in meta.items() if k in ("file_name", "page_number", "split_id")},
    )


def from_web_result(result: dict[str, Any]) -> NumberedSource:
    """A source from a web search`."""
    return NumberedSource(
        title=result.get("title") or result.get("url") or "Web result",
        content=result.get("content") or result.get("snippet") or "",
        url=result.get("url"),
        kind="web",
    )


class SourceFormatter(Protocol):
    """Renders a tool's sources as the text the answering model reads."""

    def render(self, sources: list[NumberedSource]) -> str: ...


class SourcesFormatter:
    """Sources as plain blocks."""

    def render(self, sources: list[NumberedSource]) -> str:
        return "\n\n".join(
            f"### {s.title}" + (f" ({s.url})" if s.url else "") + f"\n{s.content}" for s in sources
        )
