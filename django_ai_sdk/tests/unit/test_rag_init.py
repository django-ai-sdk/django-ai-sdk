"""
Unit tests for RAG initialization and configuration.

These tests verify RAG provider structure, configuration models, and schemas
without requiring actual retrieval operations or database setup.
"""

import pytest


class TestRAGProviderInit:
    """Test RAGProvider initialization without retrieval."""

    def test_provider_initializes_with_empty_cache(self):
        """Verify provider starts with empty cache."""
        from django_ai_sdk.rags import RAGProvider

        provider = RAGProvider()
        assert provider._cache == {}

    def test_provider_has_required_attributes(self):
        """Verify provider has cache attribute."""
        from django_ai_sdk.rags import RAGProvider

        provider = RAGProvider()
        assert hasattr(provider, "_cache")
        assert isinstance(provider._cache, dict)

    def test_provider_creates_successfully(self):
        """Verify RAGProvider can be created without errors."""
        from django_ai_sdk.rags import RAGProvider

        # Should not raise any exceptions
        provider = RAGProvider()
        assert provider is not None
        assert provider._cache == {}


class TestBaseRAGProviderStructure:
    """Test BaseRAGProvider abstract structure."""

    def test_base_is_abstract(self):
        """Verify BaseRAGProvider cannot be instantiated."""
        from django_ai_sdk.rags import BaseRAGProvider

        with pytest.raises(TypeError) as exc_info:
            BaseRAGProvider()

        assert "abstract" in str(exc_info.value).lower()

    def test_base_has_abstract_methods(self):
        """Verify abstract methods defined."""
        from django_ai_sdk.rags import BaseRAGProvider

        abstract_methods = getattr(BaseRAGProvider, "__abstractmethods__", set())
        required = {
            "warmup",
            "get_rag_instance",
            "build_tool",
            "clear_cache",
            "reindex",
        }
        assert required <= abstract_methods, (
            f"Missing abstract methods: {required - abstract_methods}"
        )


class TestRAGConfiguration:
    """Test RAG configuration models."""

    def test_rag_config_defaults(self):
        """Test RAGConfig default values."""
        from django_ai_sdk.rags import RAGConfig

        config = RAGConfig()
        assert config.top_k == 3
        assert config.document_threshold == 0.7
        assert config.embedder_model == "intfloat/multilingual-e5-large-instruct"

    def test_rag_config_custom_values(self):
        """Test RAGConfig accepts custom values."""
        from django_ai_sdk.rags import RAGConfig

        config = RAGConfig(top_k=5, document_threshold=0.8)
        assert config.top_k == 5
        assert config.document_threshold == 0.8

    def test_rag_config_custom_embedder(self):
        """Test RAGConfig accepts custom embedder model."""
        from django_ai_sdk.rags import RAGConfig

        config = RAGConfig(embedder_model="custom-model-v1")
        assert config.embedder_model == "custom-model-v1"

    def test_bm25_config_defaults(self):
        """Test BM25Config defaults exist."""
        from django_ai_sdk.rags import BM25Config

        config = BM25Config()
        assert hasattr(config, "top_k")

    def test_bm25_config_top_k_validation(self):
        """Test BM25Config validates top_k is positive."""
        from django_ai_sdk.rags import BM25Config

        # Should work with valid values
        config = BM25Config(top_k=5)
        assert config.top_k == 5


class TestRAGSchemas:
    """Test RAG Pydantic schemas."""

    def test_rag_source_creation(self):
        """Test RAGSource model."""
        from django_ai_sdk.rags import RAGSource

        source = RAGSource(id="doc1", content="test content")
        assert source.id == "doc1"
        assert source.content == "test content"
        assert source.metadata == {}

    def test_rag_source_with_metadata(self):
        """Test RAGSource with metadata."""
        from django_ai_sdk.rags import RAGSource

        source = RAGSource(
            id="doc1",
            content="content",
            metadata={"title": "Test Doc", "score": 0.95},
        )
        assert source.metadata["title"] == "Test Doc"
        assert source.metadata["score"] == 0.95

    def test_rag_source_validation_requires_id(self):
        """Test RAGSource requires id field."""
        from django_ai_sdk.rags import RAGSource

        with pytest.raises(ValueError):
            RAGSource(content="test")

    def test_rag_source_validation_requires_content(self):
        """Test RAGSource requires content field."""
        from django_ai_sdk.rags import RAGSource

        with pytest.raises(ValueError):
            RAGSource(id="doc1")

    def test_rag_result_creation(self):
        """Test RAGResult model."""
        from django_ai_sdk.rags import RAGResult, RAGSource

        result = RAGResult(
            documents=[{"id": "1", "content": "test"}],
            context="Context here",
            sources=[RAGSource(id="1", content="test")],
            query="test query",
        )
        assert result.query == "test query"
        assert len(result.sources) == 1
        assert len(result.documents) == 1
        assert result.context == "Context here"

    def test_rag_result_empty_sources(self):
        """Test RAGResult with empty sources."""
        from django_ai_sdk.rags import RAGResult

        result = RAGResult(
            documents=[],
            context="",
            sources=[],
            query="test",
        )
        assert result.sources == []
        assert result.documents == []


class TestHaystackRAGProviderStructure:
    """Test HaystackRAGProvider initialization (no Haystack setup)."""

    def test_haystack_provider_inherits_base(self):
        """Verify HaystackRAGProvider extends BaseRAGProvider."""
        from django_ai_sdk.rags import BaseRAGProvider
        from django_ai_sdk.rags.haystack import HaystackRAGProvider

        assert issubclass(HaystackRAGProvider, BaseRAGProvider)

    def test_haystack_provider_has_required_methods(self):
        """Verify HaystackRAGProvider implements abstract methods."""
        from django_ai_sdk.rags.haystack import HaystackRAGProvider

        required = ["warmup", "get_rag_instance", "build_tool", "clear_cache", "reindex"]
        for method in required:
            assert hasattr(HaystackRAGProvider, method), f"Missing method: {method}"

    def test_haystack_provider_is_not_abstract(self):
        """Verify HaystackRAGProvider can be instantiated."""
        from django_ai_sdk.rags.haystack import HaystackRAGProvider

        # Should not raise TypeError (concrete implementation)
        provider = HaystackRAGProvider()
        assert provider is not None


class TestAssistantRAGConfig:
    """Test Assistant configuration with RAG (no actual RAG calls)."""

    def test_assistant_has_rag_provider_attribute(self):
        """Verify Assistant can have rag_provider."""
        from django_ai_sdk import Assistant

        assert hasattr(Assistant, "rag_provider")

    def test_assistant_rag_provider_is_none_by_default(self):
        """Verify RAG is disabled by default."""
        from django_ai_sdk import Assistant

        assert Assistant.rag_provider is None

    def test_assistant_can_set_rag_provider(self):
        """Verify rag_provider can be set."""
        from django_ai_sdk import Assistant
        from django_ai_sdk.rags import RAGProvider

        class TestAssistant(Assistant):
            name = "Test"
            rag_provider = RAGProvider()

        assert TestAssistant.rag_provider is not None
        assert isinstance(TestAssistant.rag_provider, RAGProvider)

    def test_assistant_can_set_haystack_rag_provider(self):
        """Verify HaystackRAGProvider can be set."""
        from django_ai_sdk import Assistant
        from django_ai_sdk.rags.haystack import HaystackRAGProvider

        class TestAssistant(Assistant):
            name = "Test"
            rag_provider = HaystackRAGProvider()

        assert isinstance(TestAssistant.rag_provider, HaystackRAGProvider)


class TestRAGExports:
    """Test RAG module exports."""

    def test_all_exports_present(self):
        """Verify all expected exports in rags/__init__.py."""
        from django_ai_sdk import rags

        expected = [
            "BaseRAGAdapter",
            "BaseRAGProvider",
            "RAGProvider",
            "RAGConfig",
            "RAGResult",
            "RAGSource",
            "BM25RAG",
            "BM25Config",
            "RagDocument",
        ]

        for export in expected:
            assert hasattr(rags, export), f"Missing export: {export}"

    def test_haystack_exports_present(self):
        """Verify Haystack exports available."""
        from django_ai_sdk.rags import haystack

        assert hasattr(haystack, "HaystackRAGProvider")
