import pytest


class TestRAGProviderStructure:
    def test_provider_has_required_methods(self):
        from django_ai_sdk.rags import RAGProvider

        required = ["warmup", "get_rag_instance", "build_tool", "clear_cache", "reindex"]
        for method in required:
            assert hasattr(RAGProvider, method), f"Missing method: {method}"

    def test_provider_is_not_abstract(self):
        from django_ai_sdk.rags import RAGProvider

        provider = RAGProvider()
        assert provider is not None

    def test_provider_initializes_with_empty_cache(self):
        from django_ai_sdk.rags import RAGProvider

        provider = RAGProvider()
        assert provider._cache == {}


class TestAgentRAGConfig:
    def test_agent_has_rag_provider_attribute(self):
        from django_ai_sdk import Agent

        assert hasattr(Agent, "rag_provider")

    def test_agent_rag_provider_is_none_by_default(self):
        from django_ai_sdk import Agent

        assert Agent.rag_provider is None

    def test_agent_can_set_rag_provider(self):
        from django_ai_sdk import Agent
        from django_ai_sdk.rags import RAGProvider

        class TestAgent(Agent):
            name = "Test"
            rag_provider = RAGProvider()

        assert isinstance(TestAgent.rag_provider, RAGProvider)


class TestRAGExports:
    def test_exports_present(self):
        from django_ai_sdk import rags

        expected = [
            "RAGProvider",
            "RAGBase",
            "BM25QueryExpanderRAG",
            "BM25QueryExpanderRAGConfig",
            "ChromaDBQueryExpanderRAG",
            "ChromaDBQueryExpanderRAGConfig",
            "QdrantBM25HybridRAG",
            "QdrantBM25HybridRAGConfig",
            "RagDocument",
            "ToolSpec",
            "queryset_to_rag_documents",
            "to_document",
        ]

        for export in expected:
            assert hasattr(rags, export), f"Missing export: {export}"
