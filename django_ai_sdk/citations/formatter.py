from typing import Protocol

from pydantic import BaseModel, Field

from django_ai_sdk.logger import get_logger

logger = get_logger(__name__)


class NumberedSource(BaseModel):
    """A retrieved document assigned a citation index for a single LLM turn."""

    index: int
    title: str
    content: str
    metadata: dict = Field(default_factory=dict)


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

    RAG_TEMPLATE = (
        'Retrieved documents below are wrapped in <source id="N"> tags. '
        "When you reference one, cite inline as [N] using the exact id from "
        "the tag. Do not renumber.\n\n"
        "Citation rules:\n"
        "- Use ASCII square brackets only ([]), never CJK or fullwidth variants.\n"
        "- Place [N] immediately after the clause it supports, not bundled at "
        "the end of a paragraph.\n"
        "- Multiple sources: separate brackets with a space, e.g. [1] [2]. "
        "Never [1,2] or [1][2].\n"
        "- Do not add a 'Sources:' or 'References:' section - citations are inline only."
    )

    def format(self, documents: list[dict], start_index: int) -> tuple[str, list[NumberedSource]]:
        sources: list[NumberedSource] = []
        lines: list[str] = [self.RAG_TEMPLATE]

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
            # If document was split into chunks, add page/section marker to title.
            # Helps user distinguish between [1] policy.pdf·p2 vs [2] policy.pdf·p5.
            # split_id is the chunk index (0-indexed, so §1 means first chunk).
            split_id = meta.get("split_id")
            if split_id is not None:
                page = meta.get("page_number")
                title = f"{base} · p{page}" if page else f"{base} · §{split_id + 1}"
            else:
                title = base
            content = doc.get("content") or ""
            metadata_dict = {
                k: v for k, v in meta.items() if k in ("file_name", "page_number", "split_id")
            }
            sources.append(
                NumberedSource(index=idx, title=title, content=content, metadata=metadata_dict)
            )
            lines.append(f'<source id="{idx}">\nTitle: {title}\n{content}\n</source>')
        return "\n".join(lines), sources
