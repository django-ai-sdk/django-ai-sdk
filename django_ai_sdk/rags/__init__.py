from __future__ import annotations

from importlib import import_module
from typing import Any

from django_ai_sdk.rags.base import RAGBase, RAGConfig
from django_ai_sdk.rags.bm25 import (
    BM25QueryExpanderRAG,
    BM25QueryExpanderRAGConfig,
)
from django_ai_sdk.rags.components import MultiQueryBM25Retriever
from django_ai_sdk.rags.provider import RAGProvider
from django_ai_sdk.rags.schemas import RagDocument, ToolSpec
from django_ai_sdk.rags.utils import (
    queryset_to_rag_documents,
    to_document,
)

_LAZY = {
    "ChromaDBQueryExpanderRAG": "django_ai_sdk.rags.chroma",
    "ChromaDBQueryExpanderRAGConfig": "django_ai_sdk.rags.chroma",
    "MultiQueryChromaRetriever": "django_ai_sdk.rags.chroma",
    "QdrantBM25HybridRAG": "django_ai_sdk.rags.qdrant_hybrid",
    "QdrantBM25HybridRAGConfig": "django_ai_sdk.rags.qdrant_hybrid",
    "MultiQueryQdrantHybridRetriever": "django_ai_sdk.rags.qdrant_hybrid",
}


def __getattr__(name: str) -> Any:
    if module := _LAZY.get(name):
        return getattr(import_module(module), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BM25QueryExpanderRAG",
    "BM25QueryExpanderRAGConfig",
    "ChromaDBQueryExpanderRAG",
    "ChromaDBQueryExpanderRAGConfig",
    "MultiQueryBM25Retriever",
    "MultiQueryChromaRetriever",
    "MultiQueryQdrantHybridRetriever",
    "QdrantBM25HybridRAG",
    "QdrantBM25HybridRAGConfig",
    "RAGBase",
    "RAGConfig",
    "RAGProvider",
    "RagDocument",
    "ToolSpec",
    "queryset_to_rag_documents",
    "to_document",
]
