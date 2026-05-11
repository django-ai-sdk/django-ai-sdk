from .formatter import NumberedSource


class CitationRegistry:
    """Per-stream citation state.

    One instance is created at the start of each adapter.stream() call and
    threaded into RAG tool wrappers. Tracks a monotonic counter so multiple
    retrievals within the same turn produce non-overlapping [N] indices,
    and keeps the ordered list of sources for downstream SourceEvent emission.
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
