from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .formatter import NumberedSource, SourceFormatter, SourcesFormatter, from_document
from .registry import current_registry

if TYPE_CHECKING:
    from collections.abc import Callable

    from haystack.tools import Tool

    from .registry import SourceRegistry


def collect_sources(
    tool: Tool,
    registry: SourceRegistry | None = None,
    *,
    formatter: SourceFormatter | None = None,
    key: str | None = "documents",
    to_source: Callable[[Any], NumberedSource] = from_document,
) -> Tool:
    """Register what `tool` returns as sources of this turn; the model reads them as text"""
    formatter = formatter or SourcesFormatter()

    def _handler(items: Any) -> str:
        found = [to_source(item) for item in items or []]
        if not found:
            return "No results."
        turn = registry if registry is not None else current_registry.get()
        return formatter.render(turn.add(found) if turn is not None else found)

    tool.outputs_to_string = {"handler": _handler} | ({"source": key} if key else {})
    return tool
