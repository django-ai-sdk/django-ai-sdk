from typing import Protocol

from pydantic import BaseModel

from django_ai_sdk.logger import get_logger

logger = get_logger(__name__)


class NumberedSource(BaseModel):
    """A retrieved document assigned a citation index for a single LLM turn."""

    index: int
    title: str
    content: str


class CitationFormatter(Protocol):
    """Formats retrieved documents into LLM-visible text with inline citation markers.

    Implementations must produce a string containing [N] markers matching the
    index of each returned NumberedSource. The LLM is instructed (via the
    formatter's preamble) to reuse those exact numbers when citing.
    """

    def format(self, documents: list[dict], start_index: int) -> tuple[str, list[NumberedSource]]:
        """Return (llm_visible_string, ordered_sources).

        start_index is the first citation number to assign — callers pass the
        registry's next_index so numbering stays cumulative across multiple
        retrievals within the same turn.
        """
        ...


class DefaultCitationFormatter:
    """XML-tagged numbered citation format. Works with any chat LLM.

    The default PREAMBLE defines only the *format contract* — what the
    <source id="N"> tags mean and how to cite them. Strong models (GPT-4,
    Claude, etc.) follow this on its own.

    For weaker models that drift, subclass and extend PREAMBLE with
    STRICT_RULES (or your own).
    """

    FORMAT_CONTRACT = (
        'Retrieved documents below are wrapped in <source id="N"> tags. '
        "When you reference one, cite inline as [N] using the exact id from "
        "the tag. Do not renumber."
    )

    STRICT_RULES = (
        "Citation rules:\n"
        "- Use ASCII square brackets only ([]), never CJK or fullwidth variants.\n"
        "- Place [N] immediately after the clause it supports, not bundled at "
        "the end of a paragraph.\n"
        "- Multiple sources: separate brackets with a space, e.g. [1] [2]. "
        "Never [1,2] or [1][2].\n"
        "- Do not add a 'Sources:' or 'References:' section - citations are inline only."
    )

    PREAMBLE = FORMAT_CONTRACT

    def format(self, documents: list[dict], start_index: int) -> tuple[str, list[NumberedSource]]:
        sources: list[NumberedSource] = []
        lines: list[str] = [self.PREAMBLE]

        for offset, doc in enumerate(documents):
            idx = start_index + offset
            meta = doc.get("meta") or {}
            logger.debug("doc[%d] meta=%r", idx, meta)
            base = (
                meta.get("file_name")
                or meta.get("filename")
                or meta.get("name")
                or meta.get("title")
                or meta.get("topic")
                or f"Document {idx}"
            )
            # Disambiguate chunks from the same source file.
            # split_id is set by Haystack's document splitter (0-indexed).
            split_id = meta.get("split_id")
            if split_id is not None:
                page = meta.get("page_number")
                title = f"{base} · p{page}" if page else f"{base} · §{split_id + 1}"
            else:
                title = base
            content = doc.get("content") or ""
            sources.append(NumberedSource(index=idx, title=title, content=content))
            lines.append(f'<source id="{idx}">\nTitle: {title}\n{content}\n</source>')
        return "\n".join(lines), sources
