"""
Tests for RAG storage configuration classes.

These tests verify the config classes without any LLM calls.
"""

import pytest
from django_ai_sdk.rags.config import (
    BaseStorageConfig,
    QdrantStorageConfig,
    ChromaStorageConfig,
    BM25StorageConfig,
)


class TestBaseStorageConfig:
    """Test BaseStorageConfig."""

    def test_default_mode_is_memory(self):
        """Verify default mode is :memory:\"."""
        config = BaseStorageConfig()
        assert config.mode == ":memory:"
        assert not config.is_persistent
        assert config.persist_path is None

    def test_persistent_mode(self):
        """Verify persistent mode detection."""
        config = BaseStorageConfig(mode="persistent", persist_path="/tmp/test")
        assert config.is_persistent

    def test_memory_mode_not_persistent(self):
        """Verify :memory: mode is not persistent."""
        config = BaseStorageConfig(mode=":memory:")
        assert not config.is_persistent


class TestQdrantStorageConfig:
    """Test QdrantStorageConfig."""

    def test_defaults(self):
        """Verify default values."""
        config = QdrantStorageConfig()
        assert config.qdrant_on_disk is True
        assert config.qdrant_similarity == "cosine"
        assert config.mode == ":memory:"

    def test_custom_values(self):
        """Verify custom values are accepted."""
        config = QdrantStorageConfig(
            qdrant_on_disk=False,
            qdrant_similarity="dot",
            mode="persistent",
            persist_path="/tmp/qdrant",
        )
        assert config.qdrant_on_disk is False
        assert config.qdrant_similarity == "dot"
        assert config.is_persistent

    def test_from_settings_no_memory_id(self):
        """Verify in-memory config when no memory_id provided."""
        config = QdrantStorageConfig.from_settings(memory_id=None)
        assert config.mode == ":memory:"

    def test_from_settings_with_memory_id(self):
        """Verify persistent config when memory_id provided."""
        config = QdrantStorageConfig.from_settings(memory_id="test-123")
        if config.is_persistent:
            assert "qdrant/test-123" in config.persist_path


class TestChromaStorageConfig:
    """Test ChromaStorageConfig."""

    def test_defaults(self):
        """Verify default values."""
        config = ChromaStorageConfig()
        assert config.chroma_distance == "cosine"
        assert config.chroma_anonymized_telemetry is False
        assert config.mode == ":memory:"

    def test_custom_values(self):
        """Verify custom values are accepted."""
        config = ChromaStorageConfig(
            chroma_distance="euclidean",
            chroma_anonymized_telemetry=True,
        )
        assert config.chroma_distance == "euclidean"
        assert config.chroma_anonymized_telemetry is True

    def test_from_settings_no_memory_id(self):
        """Verify in-memory config when no memory_id provided."""
        config = ChromaStorageConfig.from_settings(memory_id=None)
        assert config.mode == ":memory:"

    def test_from_settings_with_memory_id(self):
        """Verify persistent config when memory_id provided."""
        config = ChromaStorageConfig.from_settings(memory_id="test-456")
        if config.is_persistent:
            assert "chroma/test-456" in config.persist_path


class TestBM25StorageConfig:
    """Test BM25StorageConfig."""

    def test_defaults(self):
        """Verify default values."""
        config = BM25StorageConfig()
        assert config.mode == ":memory:"

    def test_from_settings_no_memory_id(self):
        """Verify in-memory config when no memory_id provided."""
        config = BM25StorageConfig.from_settings(memory_id=None)
        assert config.mode == ":memory:"

    def test_from_settings_with_memory_id(self):
        """Verify persistent config when memory_id provided."""
        config = BM25StorageConfig.from_settings(memory_id="test-789")
        if config.is_persistent:
            assert "bm25/test-789" in config.persist_path
