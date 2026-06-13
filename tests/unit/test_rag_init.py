"""
Unit tests for RAG initialization and configuration.

Tests verify HaystackRAGProvider structure and schemas
without requiring actual retrieval operations or database setup.
"""

import pytest


class TestHaystackRAGProviderStructure:
    """Test HaystackRAGProvider initialization (no Haystack setup)."""

    def test_haystack_provider_has_required_methods(self):
        """Verify HaystackRAGProvider implements expected methods."""
        from django_ai_sdk.rags import HaystackRAGProvider

        required = ["warmup", "get_rag_instance", "build_tool", "clear_cache", "reindex"]
        for method in required:
            assert hasattr(HaystackRAGProvider, method), f"Missing method: {method}"

    def test_haystack_provider_is_not_abstract(self):
        """Verify HaystackRAGProvider can be instantiated."""
        from django_ai_sdk.rags import HaystackRAGProvider

        provider = HaystackRAGProvider()
        assert provider is not None

    def test_haystack_provider_initializes_with_empty_cache(self):
        """Verify provider starts with empty cache."""
        from django_ai_sdk.rags import HaystackRAGProvider

        provider = HaystackRAGProvider()
        assert provider._cache == {}


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

    def test_assistant_can_set_haystack_rag_provider(self):
        """Verify HaystackRAGProvider can be set."""
        from django_ai_sdk import Assistant
        from django_ai_sdk.rags import HaystackRAGProvider

        class TestAssistant(Assistant):
            name = "Test"
            rag_provider = HaystackRAGProvider()

        assert isinstance(TestAssistant.rag_provider, HaystackRAGProvider)


class TestRAGExports:
    """Test RAG module exports."""

    def test_haystack_exports_present(self):
        """Verify Haystack exports available from top-level rags package."""
        from django_ai_sdk import rags

        expected = [
            "HaystackRAGProvider",
            "HaystackRAGBase",
            "BM25QueryExpanderRAG",
            "BM25QueryExpanderRAGConfig",
            "ChromaDBQueryExpanderRAG",
            "ChromaDBQueryExpanderRAGConfig",
            "QdrantBM25HybridRAG",
            "QdrantBM25HybridRAGConfig",
            "RagDocument",
            "ToolSpec",
            "queryset_to_rag_documents",
            "rag_document_to_haystack",
        ]

        for export in expected:
            assert hasattr(rags, export), f"Missing export: {export}"
