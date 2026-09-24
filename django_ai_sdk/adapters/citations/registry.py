from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .formatter import NumberedSource


class CitationRegistry:
    """Tracks numbered citations within a single message (turn).

    When a user sends a message, a fresh registry is created. If the LLM
    calls 3 RAG tools in sequence, the registry ensures indices don't overlap:
    - Tool 1 retrieves docs → gets [1], [2], [3]
    - Tool 2 retrieves docs → gets [4], [5]

    This monotonic counter ensures the LLM cites consistently across all
    retrievals in one turn, then resets to [1] for the next message.
    """

    def __init__(self) -> None:
        self._counter = 0
        self._sources: list[NumberedSource] = []

    @property
    def next_index(self) -> int:
        return self._counter + 1

    def add(self, sources: list[NumberedSource]) -> list[NumberedSource]:
        self._sources.extend(sources)
        self._counter += len(sources)
        return sources

    @property
    def all_sources(self) -> list[NumberedSource]:
        return list(self._sources)

    def __len__(self) -> int:
        return self._counter
