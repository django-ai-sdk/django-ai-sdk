from __future__ import annotations

import hashlib
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .formatter import NumberedSource

# seed limit for sources from earlier turns
SEED_LIMIT = 30

# registry of the turn being streamed.
current_registry: ContextVar[SourceRegistry | None] = ContextVar("source_registry", default=None)


def source_key(source: NumberedSource) -> str:
    """A key that stays the same for the same source in every turn of a thread."""
    if source.url:
        return f"url:{source.url}"
    if source.doc_id:
        return f"doc:{source.doc_id}:{source.chunk_id or ''}"
    if source.chunk_id:
        return f"chunk:{source.chunk_id}"
    digest = hashlib.sha256(f"{source.title}\n{source.content}".encode()).hexdigest()[:16]
    return f"text:{digest}"


class SourceRegistry:
    """The sources an answer may cite"""

    def __init__(self) -> None:
        self._by_key: dict[str, NumberedSource] = {}
        self.seeded: set[str] = set()  # keys from earlier turns, not found again this turn

    def add(self, sources: list[NumberedSource]) -> list[NumberedSource]:
        """Register sources found this turn; returns them as registered"""
        registered = []
        for source in sources:
            key = source.key or source_key(source)
            known = self._by_key.get(key)
            if known is not None:
                known.content = source.content or known.content  # the latest text wins
                self.seeded.discard(key)
                registered.append(known)
                continue
            numbered = source.model_copy(update={"key": key, "index": len(self._by_key) + 1})
            self._by_key[key] = numbered
            registered.append(numbered)
        return registered

    def seed(self, sources: list[NumberedSource]) -> None:
        """Add sources the answer may cite without finding them this turn"""
        for source in sources:
            key = source.key or source_key(source)
            if key not in self._by_key:
                self.add([source.model_copy(update={"key": key})])
                self.seeded.add(key)

    async def seed_from_thread(self, thread_id: str, limit: int = SEED_LIMIT) -> None:
        """Seed with the sources stored in the thread's citation artifacts."""
        from django_ai_sdk.artifacts.models import Artifact  # noqa: PLC0415
        from django_ai_sdk.artifacts.schemas import (  # noqa: PLC0415
            CitationsArtifact,
            CitedSource,
        )

        from .formatter import NumberedSource  # noqa: PLC0415

        found: dict[str, NumberedSource] = {}
        rows = Artifact.objects.filter(
            thread_id=thread_id, schema_name=CitationsArtifact.__name__
        ).order_by("-created_at")
        async for row in rows:
            # only the sources
            for s in map(CitedSource.model_validate, row.data.get("sources", [])):
                if s.key not in found and len(found) < limit:
                    found[s.key] = NumberedSource(
                        key=s.key,
                        title=s.title,
                        content=s.content,
                        kind=s.kind,
                        url=s.url,
                        doc_id=s.document_id,
                        chunk_id=s.chunk_id,
                    )
        self.seed(list(found.values()))

    def get(self, number: int) -> NumberedSource | None:
        return next((s for s in self._by_key.values() if s.index == number), None)

    @property
    def all(self) -> list[NumberedSource]:
        return list(self._by_key.values())

    def __len__(self) -> int:
        return len(self._by_key)
