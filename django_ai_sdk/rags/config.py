from pydantic import BaseModel, Field
from typing import Literal
from django.conf import settings


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
        """Create config from Django settings."""
        persist_path = getattr(settings, "AI_SDK_VECTOR_STORE_PATH", None)

        if persist_path and silo_id:
            return cls(mode="persistent", persist_path=f"{persist_path}/qdrant/{silo_id}")
        elif persist_path:
            return cls(mode="persistent", persist_path=f"{persist_path}/qdrant/default")

        return cls(mode=":memory:")
