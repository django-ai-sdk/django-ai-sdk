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
    """Wire a RAG ComponentTool so its LLM-visible string carries [N] markers.

    The tool's structured output (the documents list) is preserved for
    downstream consumers (adapter source-event emission, storage, etc.). Only
    the string the LLM consumes is rewritten.

    Each invocation increments the registry's counter so multiple retrievals
    within the same turn produce non-overlapping indices.
    """

    def _handler(documents: Any) -> str:
        as_dicts = []
        for d in documents or []:
            if hasattr(d, "meta"):
                as_dicts.append(
                    {
                        "id": getattr(d, "id", None),
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
