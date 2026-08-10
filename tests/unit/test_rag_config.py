"""
Tests for RAG storage configuration classes.

These tests verify the config classes without any LLM calls.
"""

from django.test.utils import override_settings

from django_ai_sdk.rags.config import (
    BaseStorageConfig,
    QdrantStorageConfig,
    ChromaStorageConfig,
    BM25StorageConfig,
)


class TestBaseStorageConfig:
    """Test BaseStorageConfig."""

    def test_default_backend_is_memory(self):
        """Verify default backend is :memory:."""
        config = BaseStorageConfig()
        assert config.backend == ":memory:"
        assert not config.is_persistent
        assert config.persist_path is None

    def test_persistent_backend(self):
        """Verify persistent backend detection."""
        config = BaseStorageConfig(backend="persistent", persist_path="/tmp/test")
        assert config.is_persistent

    def test_memory_backend_not_persistent(self):
        """Verify :memory: backend is not persistent."""
        config = BaseStorageConfig(backend=":memory:")
        assert not config.is_persistent


class TestQdrantStorageConfig:
    """Test QdrantStorageConfig."""

    def test_defaults(self):
        """Verify default values."""
        config = QdrantStorageConfig()
        assert config.similarity == "cosine"
        assert config.backend == ":memory:"

    def test_custom_values(self):
        """Verify custom values are accepted."""
        config = QdrantStorageConfig(
            similarity="dot",
            backend="persistent",
            persist_path="/tmp/qdrant",
        )
        assert config.similarity == "dot"
        assert config.is_persistent

    def test_from_settings_no_memory_id(self):
        """Verify in-memory config when no memory_id provided."""
        with override_settings(AI_SDK_VECTOR_STORE_URL=None, AI_SDK_VECTOR_STORE_PATH=None):
            config = QdrantStorageConfig.from_settings(memory_id=None)
            assert config.backend == ":memory:"

    def test_from_settings_with_memory_id(self):
        """Verify persistent config when memory_id provided."""
        config = QdrantStorageConfig.from_settings(memory_id="test-123")
        if config.is_persistent:
            assert "qdrant/test-123" in config.persist_path

    def test_from_settings_kwargs_passthrough(self):
        """Verify extra kwargs are captured and accessible via .extra."""
        with override_settings(AI_SDK_VECTOR_STORE_URL=None, AI_SDK_VECTOR_STORE_PATH=None):
            config = QdrantStorageConfig.from_settings(
                memory_id=None, prefer_grpc=False, timeout=30
            )
            assert config.extra["prefer_grpc"] is False
            assert config.extra["timeout"] == 30

    def test_from_settings_server_index_derived_from_memory_id(self):
        """Verify index is derived from memory_id in server mode."""
        config = QdrantStorageConfig.from_settings(memory_id="abc-123")
        if config.is_server:
            assert config.extra["index"] == "memory_abc-123"

    def test_from_settings_server_index_override(self):
        """Verify user can override index via kwargs."""
        config = QdrantStorageConfig.from_settings(memory_id="abc", index="custom")
        if config.is_server:
            assert config.extra["index"] == "custom"


class TestChromaStorageConfig:
    """Test ChromaStorageConfig."""

    def test_defaults(self):
        """Verify default values."""
        config = ChromaStorageConfig()
        assert config.backend == ":memory:"

    def test_custom_values(self):
        """Verify custom values are accepted via extra kwargs."""
        config = ChromaStorageConfig(distance_function="euclidean")
        assert config.extra["distance_function"] == "euclidean"

    def test_from_settings_no_memory_id(self):
        """Verify in-memory config when no memory_id provided."""
        with override_settings(AI_SDK_VECTOR_STORE_PATH=None):
            config = ChromaStorageConfig.from_settings(memory_id=None)
            assert config.backend == ":memory:"

    def test_from_settings_with_memory_id(self):
        """Verify persistent config when memory_id provided."""
        config = ChromaStorageConfig.from_settings(memory_id="test-456")
        if config.is_persistent:
            assert "chroma/test-456" in config.persist_path

    def test_from_settings_kwargs_passthrough(self):
        """Verify extra kwargs are captured."""
        with override_settings(AI_SDK_VECTOR_STORE_PATH=None):
            config = ChromaStorageConfig.from_settings(
                memory_id=None, distance_function="cosine"
            )
            assert config.extra["distance_function"] == "cosine"


class TestBM25StorageConfig:
    """Test BM25StorageConfig."""

    def test_defaults(self):
        """Verify default values."""
        config = BM25StorageConfig()
        assert config.backend == ":memory:"

    def test_from_settings_no_memory_id(self):
        """Verify in-memory config when no memory_id provided."""
        with override_settings(AI_SDK_VECTOR_STORE_PATH=None):
            config = BM25StorageConfig.from_settings(memory_id=None)
            assert config.backend == ":memory:"

    def test_from_settings_with_memory_id(self):
        """Verify persistent config when memory_id provided."""
        config = BM25StorageConfig.from_settings(memory_id="test-789")
        if config.is_persistent:
            assert "bm25/test-789" in config.persist_path
