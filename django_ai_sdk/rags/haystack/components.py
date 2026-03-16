from typing import Any

from haystack import component
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import Document as HaystackDocument
from haystack_integrations.components.retrievers.chroma import ChromaQueryTextRetriever
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever


@component
class MultiQueryChromaRetriever:
    """Retriever that runs multiple queries and deduplicates results for ChromaDB"""

    def __init__(
        self,
        document_store: Any,
        top_k: int = 3,
    ) -> None:
        self.document_store = document_store
        self.top_k = top_k
        self.retriever = ChromaQueryTextRetriever(
            document_store=document_store,
            top_k=top_k,
        )

    @component.output_types(documents=list[HaystackDocument])
    def run(
        self, queries: list[str], top_k: int | None = None
    ) -> dict[str, list[HaystackDocument]]:
        if top_k is not None:
            self.top_k = top_k

        all_docs = {}
        for query in queries:
            result = self.retriever.run(query=query, top_k=self.top_k)
            for doc in result["documents"]:
                all_docs[doc.id] = doc

        all_docs = list(all_docs.values())
        all_docs.sort(key=lambda x: x.score if hasattr(x, "score") else 0, reverse=True)
        return {"documents": all_docs}


@component
class MultiQueryBM25Retriever:
    """Retriever that runs multiple queries and deduplicates results for BM25."""

    def __init__(
        self,
        document_store: Any,
        top_k: int = 3,
    ) -> None:
        self.document_store = document_store
        self.top_k = top_k
        self.retriever = InMemoryBM25Retriever(document_store=document_store, top_k=top_k)

    @component.output_types(documents=list[HaystackDocument])
    def run(
        self, queries: list[str], top_k: int | None = None
    ) -> dict[str, list[HaystackDocument]]:
        if top_k is not None:
            self.top_k = top_k

        all_docs = {}
        for query in queries:
            result = self.retriever.run(query=query, top_k=self.top_k)
            for doc in result["documents"]:
                all_docs[doc.id] = doc

        all_docs = list(all_docs.values())
        all_docs.sort(key=lambda x: x.score if hasattr(x, "score") else 0, reverse=True)
        return {"documents": all_docs}


@component
class MultiQueryQdrantHybridRetriever:
    """Retriever that runs multiple queries with hybrid search and deduplicates results."""

    def __init__(
        self,
        document_store: Any,
        top_k: int = 3,
        sparse_embedder_model: str = "Qdrant/bm42-all-minilm-l6-v2-attentions",
        dense_embedder_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ) -> None:
        self.document_store = document_store
        self.top_k = top_k
        self.sparse_embedder_model = sparse_embedder_model
        self.dense_embedder_model = dense_embedder_model
        self._sparse_embedder = None
        self._dense_embedder = None

    def warm_up(self) -> None:
        from haystack_integrations.components.embedders.fastembed import (
            FastembedSparseTextEmbedder,
            FastembedTextEmbedder,
        )

        self._sparse_embedder = FastembedSparseTextEmbedder(model=self.sparse_embedder_model)
        self._sparse_embedder.warm_up()
        self._dense_embedder = FastembedTextEmbedder(model=self.dense_embedder_model)
        self._dense_embedder.warm_up()

    @component.output_types(documents=list[HaystackDocument])
    def run(
        self, queries: str | list[str], top_k: int | None = None
    ) -> dict[str, list[HaystackDocument]]:
        if self._sparse_embedder is None:
            self.warm_up()

        # Convert single query to list
        if isinstance(queries, str):
            queries = [queries]

        k = top_k or self.top_k

        all_docs = {}
        for query in queries:
            assert self._sparse_embedder is not None
            assert self._dense_embedder is not None
            sparse_result = self._sparse_embedder.run(text=query)
            dense_result = self._dense_embedder.run(text=query)

            retriever = QdrantHybridRetriever(document_store=self.document_store)
            result = retriever.run(
                query_sparse_embedding=sparse_result["sparse_embedding"],
                query_embedding=dense_result["embedding"],
                top_k=k,
            )
            for doc in result["documents"]:
                if doc.id not in all_docs:
                    all_docs[doc.id] = doc

        all_docs = list(all_docs.values())
        all_docs.sort(key=lambda x: x.score if hasattr(x, "score") else 0, reverse=True)
        return {"documents": all_docs[:k]}
