"""Bridge between the citation primitives and Haystack's `ComponentTool`.

Kept separate from `formatter.py` and `registry.py` so the core primitives stay
adapter-agnostic. Only this module imports Haystack.
"""

from typing import Any

from haystack.tools import ComponentTool

from .formatter import CitationFormatter
from .registry import CitationRegistry


def attach_citations(
    tool: ComponentTool,
    formatter: CitationFormatter,
    registry: CitationRegistry,
    documents_key: str = "documents",
) -> ComponentTool:
    """Wire citations into a RAG tool so the LLM sees [1] [2] [3] markers.

    How it works:
    1. Tool retrieves raw documents (e.g., chunks from a PDF)
    2. Handler converts them to dicts and runs through the formatter
    3. Formatter returns: XML with <source id="1"> tags for the LLM,
       plus a list of NumberedSource objects
    4. Registry stores the NumberedSource objects for persistence/emission
    5. LLM-visible string with [N] markers is returned to the tool
    6. Raw documents are preserved unchanged (for downstream processors)

    The registry counter increments so multiple RAG calls in one turn get
    non-overlapping indices: tool1 gets [1,2,3], tool2 gets [4,5].
    """

    def _handler(documents: list[Any] | None) -> str:
        # Convert Haystack Document objects to plain dicts for the formatter.
        as_dicts = []
        for d in documents or []:
            if hasattr(d, "meta"):
                as_dicts.append(
                    {
                        "chunk_id": getattr(d, "id", None),
                        "content": getattr(d, "content", "") or "",
                        "meta": dict(getattr(d, "meta", {}) or {}),
                    }
                )
            else:
                as_dicts.append(dict(d))

        text, sources = formatter.format(as_dicts, start_index=registry.next_index)
        registry.add(sources)
        return text

    tool.outputs_to_string = {"source": documents_key, "handler": _handler}
    return tool
