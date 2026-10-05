from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field


class NumberedSource(BaseModel):
    """A retrieved document assigned a citation index for a single LLM turn."""

    index: int
    title: str
    content: str
    chunk_id: str | None = None
    doc_id: str | None = None
    memory_id: str | None = None
    page_number: int | None = None
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )  # Display fields for UI (file_name, split_id)


class CitationFormatter(Protocol):
    """Formats retrieved documents into LLM-visible text with inline citation markers.

    Implementations must produce a string containing [N] markers matching the
    index of each returned NumberedSource. The LLM is instructed (via the
    formatter's preamble) to reuse those exact numbers when citing.
    """

    def format(self, documents: list[dict], start_index: int) -> tuple[str, list[NumberedSource]]:
        """Return (llm_visible_string, ordered_sources).

        start_index is the first citation number to assign - callers pass the
        registry's next_index so numbering stays cumulative across multiple
        retrievals within the same turn.
        """
        ...


class DefaultCitationFormatter:
    """XML-tagged numbered citation format. Works with any chat LLM.

    Override RAG_TEMPLATE in a subclass to customize citation instructions.
    Default includes strict rules to prevent common citation errors (fullwidth
    brackets, bundled citations, trailing 'Sources:' sections).
    """

    # Documents are wrapped in <source id="N">, so models reach for that same tag
    # when citing; the self-closing <source id="N" /> is the one citation format.
    RAG_TEMPLATE = (
        'Retrieved documents below are each wrapped in a <source id="N">...</source> tag. '
        'To reference one, cite it inline with a self-closing <source id="N" /> tag, '
        "using the exact id from the source wrapper. Do not renumber.\n\n"
        "Citation rules:\n"
        '- Use exactly this self-closing format: <source id="N" />, one tag per source.\n'
        "- Place the citation tag immediately after the clause it supports, not bundled at "
        "the end of a paragraph.\n"
        '- Multiple sources: use separate tags, e.g. <source id="1" /> <source id="2" />. '
        "Never combine ids in one tag.\n"
        "- Do not add a 'Sources:' or 'References:' section - citations are inline only.\n"
        "- Do not explain or reason about citations, only add them.\n"
        "- Answer from the sources below: they were retrieved for this question and usually "
        "hold the answer, also when they word it differently or it has to be read from a "
        "table, a code or a field definition. Only when none of them is about what the user "
        "asked (e.g. a different file than the one named), say you could not find it; never "
        "fill in from general knowledge."
    )

    # Repeated after the sources
    CITATION_REMINDER = (
        'Reminder: answer from the sources above, citing inline with <source id="N" />, '
        "one tag per source id."
    )

    def format(self, documents: list[dict], start_index: int) -> tuple[str, list[NumberedSource]]:
        if not documents:
            return "", []
        sources: list[NumberedSource] = []
        lines: list[str] = [self.RAG_TEMPLATE]
        files: dict[str, int] = {}  # passages per file, in order of first appearance

        for offset, doc in enumerate(documents):
            idx = start_index + offset
            meta: dict[str, Any] = doc.get("meta") or {}
            base = (
                meta.get("file_name")
                or meta.get("filename")
                or meta.get("name")
                or meta.get("title")
                or meta.get("topic")
                or f"Document {idx}"
            )
            files[base] = files.get(base, 0) + 1
            split_id = meta.get("split_id")
            page_number = meta.get("page_number")
            title = f"{base} · §{split_id + 1}" if split_id is not None else base
            content = doc.get("content", "")
            metadata_dict = {
                k: v for k, v in meta.items() if k in ("file_name", "page_number", "split_id")
            }
            sources.append(
                NumberedSource(
                    index=idx,
                    title=title,
                    content=content,
                    chunk_id=doc.get("chunk_id"),
                    doc_id=meta.get("doc_id"),
                    memory_id=meta.get("memory_id"),
                    page_number=page_number,
                    metadata=metadata_dict,
                )
            )
            lines.append(f'<source id="{idx}">\nTitle: {title}\n{content}\n</source>')
        lines.insert(1, self.files_line(files))
        lines.append(self.CITATION_REMINDER)
        # Its own last line, where it weighs most (4 of 5 answers then named the file,
        # 2 of 5 when it was part of the reminder): the files, not "the document".
        lines.append(f"Name the file each point comes from ({', '.join(files)}).")
        return "\n".join(lines), sources

    @staticmethod
    def files_line(files: dict[str, int]) -> str:
        """Which files the passages come from, so the answer can name them ("only
        cv.pdf mentions Java") instead of "the consultant" or "the document"."""
        listed = ", ".join(
            f"{name} ({count} passage{'s' if count > 1 else ''})" for name, count in files.items()
        )
        total = sum(files.values())
        return (
            f"These {total} passages come from {len(files)} "
            f"{'file' if len(files) == 1 else 'files'}: {listed}.\n"
        )
