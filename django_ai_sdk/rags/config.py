from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from django_ai_sdk.logger import get_logger
from django_ai_sdk.utils import resolve_setting

logger = get_logger(__name__)


class BaseStorageConfig(BaseModel):
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
        persist_path = resolve_setting("AI_SDK_VECTOR_STORE_PATH")
        if not persist_path:
            return None
        base_path = str(persist_path).rstrip("/")
        return f"{base_path}/{backend}/{memory_id}"

    @classmethod
    def from_settings(cls, memory_id: str | None = None, **kwargs: Any) -> BaseStorageConfig:
        """
        Args:
            **kwargs: Additional keyword arguments forwarded to the document store constructor.
        """
        path = cls._build_path("base", memory_id)
        if path:
            return cls(backend="persistent", persist_path=path, **kwargs)
        return cls(backend=":memory:", **kwargs)


class QdrantStorageConfig(BaseStorageConfig):
    backend: Literal[":memory:", "persistent", "server"] = Field(default=":memory:")  # ty: ignore[incompatible-attribute-override]
    location: str | None = Field(default=None)
    similarity: Literal["cosine", "dot", "euclidean"] = Field(default="cosine")
    lock_timeout: float = Field(default=10.0, ge=0)

    @property
    def is_server(self) -> bool:
        return self.backend == "server" and self.location is not None

    @classmethod
    def from_settings(cls, memory_id: str | None = None, **kwargs: Any) -> QdrantStorageConfig:
        """
        Args:
            **kwargs: Additional keyword arguments forwarded to QdrantDocumentStore.
        """
        url = resolve_setting("AI_SDK_VECTOR_STORE_URL")
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
    @classmethod
    def from_settings(cls, memory_id: str | None = None, **kwargs: Any) -> ChromaStorageConfig:
        """
        Args:
            **kwargs: Additional keyword arguments forwarded to ChromaDocumentStore.
        """
        path = cls._build_path("chroma", memory_id)
        if path:
            return cls(backend="persistent", persist_path=path, **kwargs)
        return cls(backend=":memory:", **kwargs)


class BM25StorageConfig(BaseStorageConfig):
    @classmethod
    def from_settings(cls, memory_id: str | None = None, **kwargs: Any) -> BM25StorageConfig:
        """
        Args:
            **kwargs: Additional keyword arguments forwarded to the constructor.
        """
        path = cls._build_path("bm25", memory_id)
        if path:
            return cls(backend="persistent", persist_path=path, **kwargs)
        return cls(backend=":memory:", **kwargs)
