from __future__ import annotations

from typing import Literal

from django.conf import settings
from pydantic import BaseModel, Field

from django_ai_sdk.logger import get_logger

logger = get_logger(__name__)


class BaseStorageConfig(BaseModel):
    """Base configuration for vector store persistence."""

    persist_path: str | None = Field(default=None)
    mode: Literal[":memory:", "persistent"] = Field(default=":memory:")

    @property
    def is_persistent(self) -> bool:
        return self.mode == "persistent" and self.persist_path is not None

    @classmethod
    def _build_path(cls, backend: str, memory_id: str) -> str | None:
        """Build the persistence path for a given backend and memory_id."""
        persist_path = getattr(settings, "AI_SDK_VECTOR_STORE_PATH", None)
        if not persist_path:
            return None
        base_path = str(persist_path).rstrip("/")
        return f"{base_path}/{backend}/{memory_id}"

    @classmethod
    def from_settings(cls, memory_id: str | None = None) -> BaseStorageConfig:
        """Create config from Django settings.

        Args:
            memory_id: The memory ID determines the storage path.
                       Returns in-memory config if memory_id is None or empty.
        """
        if not memory_id:
            return cls(mode=":memory:")
        return cls(mode=":memory:")


class QdrantStorageConfig(BaseStorageConfig):
    """Configuration for Qdrant vector store persistence."""

    qdrant_on_disk: bool = Field(default=True)
    qdrant_similarity: Literal["cosine", "dot", "euclidean"] = Field(default="cosine")

    @classmethod
    def from_settings(cls, memory_id: str | None = None) -> QdrantStorageConfig:
        """Create config from Django settings.

        Args:
            memory_id: The memory ID determines the storage path.
                       Returns in-memory config if memory_id is None or empty.
        """
        if not memory_id:
            return cls(mode=":memory:")

        path = cls._build_path("qdrant", memory_id)
        if path:
            return cls(mode="persistent", persist_path=path)
        return cls(mode=":memory:")


class ChromaStorageConfig(BaseStorageConfig):
    """Configuration for Chroma vector store persistence."""

    chroma_distance: str = Field(default="cosine")
    chroma_anonymized_telemetry: bool = Field(default=False)

    @classmethod
    def from_settings(cls, memory_id: str | None = None) -> ChromaStorageConfig:
        """Create config from Django settings.

        Args:
            memory_id: The memory ID determines the storage path.
                       Returns in-memory config if memory_id is None or empty.
        """
        if not memory_id:
            return cls(mode=":memory:")

        path = cls._build_path("chroma", memory_id)
        if path:
            return cls(mode="persistent", persist_path=path)
        return cls(mode=":memory:")


class BM25StorageConfig(BaseStorageConfig):
    """Configuration for BM25 (plain) persistence."""

    @classmethod
    def from_settings(cls, memory_id: str | None = None) -> BM25StorageConfig:
        """Create config from Django settings.

        Args:
            memory_id: The memory ID determines the storage path.
                       Returns in-memory config if memory_id is None or empty.
        """
        if not memory_id:
            return cls(mode=":memory:")

        path = cls._build_path("bm25", memory_id)
        if path:
            return cls(mode="persistent", persist_path=path)
        return cls(mode=":memory:")
