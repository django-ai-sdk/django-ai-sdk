from typing import Literal

from django.conf import settings
from pydantic import BaseModel, Field


class VectorDBStorageConfig(BaseModel):
    """Configuration for vector store persistence."""

    persist_path: str | None = Field(default=None)
    mode: Literal[":memory:", "persistent"] = Field(default=":memory:")

    # Chroma settings
    chroma_distance: str = Field(default="cosine")
    chroma_anonymized_telemetry: bool = Field(default=False)

    # Qdrant settings
    qdrant_on_disk: bool = Field(default=True)
    qdrant_similarity: Literal["cosine", "dot", "euclidean"] = Field(default="cosine")

    @property
    def is_persistent(self) -> bool:
        return self.mode == "persistent" and self.persist_path is not None

    @classmethod
    def from_settings(cls, silo_id: str | None = None) -> "VectorDBStorageConfig":
        """Create config from Django settings.

        Args:
            silo_id: Required. The silo ID determines the storage path.
                     Returns in-memory config if silo_id is None or empty.
        """
        if not silo_id:
            return cls(mode=":memory:")

        persist_path = getattr(settings, "AI_SDK_VECTOR_STORE_PATH", None)

        if persist_path:
            base_path = persist_path.rstrip("/")
            return cls(mode="persistent", persist_path=f"{base_path}/qdrant/{silo_id}")

        return cls(mode=":memory:")
