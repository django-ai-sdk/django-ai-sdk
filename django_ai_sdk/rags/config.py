from __future__ import annotations

from typing import Any, Literal

from django.conf import settings
from pydantic import BaseModel, ConfigDict, Field

from django_ai_sdk.logger import get_logger

logger = get_logger(__name__)


class BaseStorageConfig(BaseModel):
    """Base configuration for vector store persistence."""

    model_config = ConfigDict(extra="allow")

    persist_path: str | None = Field(default=None)
    backend: Literal[":memory:", "persistent"] = Field(default=":memory:")

    @property
    def is_persistent(self) -> bool:
        return self.backend == "persistent" and self.persist_path is not None

    @property
    def extra(self) -> dict[str, Any]:
        """Extra kwargs captured via extra='allow', forwarded to document store."""
        return self.__pydantic_extra__ or {}

    @classmethod
    def _build_path(cls, backend: str, memory_id: str | None) -> str | None:
        """Build the persistence path for a given backend and memory_id."""
        persist_path = getattr(settings, "AI_SDK_VECTOR_STORE_PATH", None)
        if not persist_path:
            return None
        base_path = str(persist_path).rstrip("/")
        return f"{base_path}/{backend}/{memory_id}"

    @classmethod
    def from_settings(cls, memory_id: str | None = None, **kwargs: Any) -> BaseStorageConfig:
        """Create config from Django settings.

        Args:
            memory_id: The memory ID determines the storage path.
            **kwargs: Additional keyword arguments forwarded to the document store
                     constructor (e.g. prefer_grpc, api_key, timeout for Qdrant).
        """
        path = cls._build_path("base", memory_id)
        if path:
            return cls(backend="persistent", persist_path=path, **kwargs)
        return cls(backend=":memory:", **kwargs)


class QdrantStorageConfig(BaseStorageConfig):
    """Configuration for Qdrant vector store persistence."""

    backend: Literal[":memory:", "persistent", "server"] = Field(default=":memory:")
    location: str | None = Field(default=None)
    similarity: Literal["cosine", "dot", "euclidean"] = Field(default="cosine")

    @property
    def is_server(self) -> bool:
        return self.backend == "server" and self.location is not None

    @classmethod
    def from_settings(cls, memory_id: str | None = None, **kwargs: Any) -> QdrantStorageConfig:
        """Create config from Django settings.

        Args:
            memory_id: The memory ID determines the storage path or collection name.
            **kwargs: Additional keyword arguments forwarded to QdrantDocumentStore
                     (e.g. prefer_grpc, https, api_key=Secret.from_token(...),
                     grpc_port, index, timeout).
        """
        url = getattr(settings, "AI_SDK_VECTOR_STORE_URL", None)
        if url:
            kwargs.setdefault("index", f"memory_{memory_id}" if memory_id else "default")
            return cls(
                backend="server",
                location=url,
                **kwargs,
            )

        path = cls._build_path("qdrant", memory_id)
        if path:
            return cls(backend="persistent", persist_path=path, **kwargs)
        return cls(backend=":memory:", **kwargs)


class ChromaStorageConfig(BaseStorageConfig):
    """Configuration for Chroma vector store persistence."""

    @classmethod
    def from_settings(cls, memory_id: str | None = None, **kwargs: Any) -> ChromaStorageConfig:
        """Create config from Django settings.

        Args:
            memory_id: The memory ID determines the storage path.
            **kwargs: Additional keyword arguments forwarded to ChromaDocumentStore
                     (e.g. distance_function, host, port).
        """
        path = cls._build_path("chroma", memory_id)
        if path:
            return cls(backend="persistent", persist_path=path, **kwargs)
        return cls(backend=":memory:", **kwargs)


class BM25StorageConfig(BaseStorageConfig):
    """Configuration for BM25 (plain) persistence."""

    @classmethod
    def from_settings(cls, memory_id: str | None = None, **kwargs: Any) -> BM25StorageConfig:
        """Create config from Django settings.

        Args:
            memory_id: The memory ID determines the storage path.
            **kwargs: Additional keyword arguments forwarded to the constructor.
        """
        path = cls._build_path("bm25", memory_id)
        if path:
            return cls(backend="persistent", persist_path=path, **kwargs)
        return cls(backend=":memory:", **kwargs)
