from typing import Any

from haystack import component
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import Document as HaystackDocument
from haystack_integrations.components.embedders.fastembed import (
    FastembedSparseTextEmbedder,
    FastembedTextEmbedder,
)
from haystack_integrations.components.retrievers.chroma import ChromaQueryTextRetriever
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever

from django_ai_sdk.logger import get_logger

logger = get_logger(__name__)


class MultiQueryDeduplicationMixin:
    """
    Mixin providing multi-query deduplication and ranking logic.

    This can be used by any retriever that needs to:
    1. Run multiple queries
    2. Deduplicate results by document ID
    3. Sort by score (descending)
    4. Return top_k results

    Example usage:
        class MyRetriever(MultiQueryDeduplicationMixin):
            def run(self, queries: list[str], top_k: int = 3):
                # Run your queries
                results = []
                for query in queries:
                    results.append(self._retrieve(query, top_k))

                # Use mixin to deduplicate and rank
                docs = self.deduplicate_and_rank(results, top_k)
                return {"documents": docs}
    """

    @staticmethod
    def deduplicate_and_rank(
        results: list[dict[str, list[HaystackDocument]]],
        top_k: int,
        min_score: float | None = None,
    ) -> list[HaystackDocument]:
        """
        Deduplicate documents across multiple query results, filter by score, and rank.

        Args:
            results: List of retrieval result dicts, each containing "documents"
            top_k: Maximum number of documents to return
            min_score: Minimum score threshold; documents below this are dropped.
                       None disables filtering.

        Returns:
            List of unique documents sorted by score (descending), limited to top_k
        """
        # Deduplicate by document ID
        all_docs: dict[str, HaystackDocument] = {}
        for result in results:
            for doc in result.get("documents", []):
                if doc.id not in all_docs:
                    all_docs[doc.id] = doc

        # Sort by score
        docs = list(all_docs.values())
        docs.sort(
            key=lambda x: x.score if hasattr(x, "score") and x.score is not None else 0.0,
            reverse=True,
        )

        # Keep filter for non native support for min_score (BM25 e.g.)
        if min_score is not None:
            docs = [d for d in docs if (d.score or 0.0) >= min_score]

        return docs[:top_k]


@component
class BaseMultiQueryRetriever(MultiQueryDeduplicationMixin):
    """
    Base class for multi-query retrievers with deduplication.

    This provides a template method pattern where subclasses only need to
    implement retrieve() for their specific retriever type.

    Suitable for simple retrievers like BM25 and Chroma that don't need
    special preprocessing (like embedding generation).

    Example:
        @component
        class MultiQueryBM25Retriever(BaseMultiQueryRetriever):
            def __init__(self, document_store, top_k=3):
                super().__init__(document_store, top_k)
                from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
                self._retriever = InMemoryBM25Retriever(
                    document_store=document_store, top_k=top_k
                )

            def retrieve(self, query: str, top_k: int) -> dict:
                return self._retriever.run(query=query, top_k=top_k)
    """

    def __init__(
        self,
        document_store: Any,
        top_k: int = 3,
        min_score: float | None = None,
    ) -> None:
        self.document_store = document_store
        self.top_k = top_k
        self.min_score = min_score
        self._retriever: Any

    def retrieve(self, query: str, top_k: int) -> dict[str, list[HaystackDocument]]:
        """
        Perform a single query retrieval.

        Subclasses must override this method to perform the actual retrieval
        using their specific retriever implementation.

        Args:
            query: The search query string
            top_k: Maximum documents to retrieve for this query

        Returns:
            Dict with "documents" key containing list of HaystackDocument
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement retrieve()")

    @component.output_types(documents=list[HaystackDocument])
    def run(
        self, queries: list[str], top_k: int | None = None
    ) -> dict[str, list[HaystackDocument]]:
        """
        Run multiple queries and return deduplicated results.

        Args:
            queries: List of query strings to execute
            top_k: Maximum documents to return (uses self.top_k if None)

        Returns:
            Dict with "documents" key containing unique documents sorted by score
        """
        k = top_k if top_k is not None else self.top_k

        # Run all queries
        results: list[dict[str, list[HaystackDocument]]] = []
        for query in queries:
            result = self.retrieve(query, k)
            results.append(result)

        docs = self.deduplicate_and_rank(results, k, self.min_score)
        return {"documents": docs}


@component
class MultiQueryChromaRetriever(BaseMultiQueryRetriever):
    """
    Retriever that runs multiple queries and deduplicates results for ChromaDB.

    Uses BaseMultiQueryRetriever to handle multi-query logic.
    Just implements retrieve() with ChromaQueryTextRetriever.

    Example:
        retriever = MultiQueryChromaRetriever(
            document_store=chroma_store,
            top_k=3
        )
        result = retriever.run(queries=["query1", "query2", "query3"])
    """

    def __init__(
        self,
        document_store: Any,
        top_k: int = 3,
    ) -> None:
        super().__init__(document_store, top_k)

        self._retriever = ChromaQueryTextRetriever(
            document_store=document_store,
            top_k=top_k,
        )

    def retrieve(self, query: str, top_k: int) -> dict[str, list[HaystackDocument]]:
        """Retrieve documents for a single query using ChromaDB."""
        return self._retriever.run(query=query, top_k=top_k)


@component
class MultiQueryBM25Retriever(BaseMultiQueryRetriever):
    """
    Retriever that runs multiple queries and deduplicates results for BM25.
    Uses BaseMultiQueryRetriever to handle multi-query logic.

    Example:
        retriever = MultiQueryBM25Retriever(
            document_store=memory_store,
            top_k=3
        )
        result = retriever.run(queries=["query1", "query2", "query3"])
    """

    def __init__(
        self,
        document_store: Any,
        top_k: int = 3,
    ) -> None:
        super().__init__(document_store, top_k)

        self._retriever = InMemoryBM25Retriever(
            document_store=document_store,
            top_k=top_k,
        )

    def retrieve(self, query: str, top_k: int) -> dict[str, list[HaystackDocument]]:
        """Retrieve documents for a single query using BM25."""
        return self._retriever.run(query=query, top_k=top_k)


@component
class MultiQueryQdrantHybridRetriever(MultiQueryDeduplicationMixin):
    """
    Retriever that runs multiple queries with hybrid search and deduplicates results.

    This retriever is more complex than Chroma/BM25 because it requires:
    1. Embedding generation (sparse + dense)
    2. Warmup for embedders
    3. Special handling for query preprocessing

    Uses MultiQueryDeduplicationMixin directly (not BaseMultiQueryRetriever)
    because it needs custom run() logic for embeddings.

    Example:
        retriever = MultiQueryQdrantHybridRetriever(
            document_store=qdrant_store,
            top_k=3
        )
        retriever.warm_up()  # Load embedders
        result = retriever.run(queries=["query1", "query2"])
    """

    def __init__(
        self,
        document_store: Any,
        top_k: int = 3,
        min_score: float | None = None,
        sparse_embedder_model: str = "Qdrant/bm42-all-minilm-l6-v2-attentions",
        dense_embedder_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ) -> None:
        self.document_store = document_store
        self.top_k = top_k
        self.min_score = min_score
        self.sparse_embedder_model = sparse_embedder_model
        self.dense_embedder_model = dense_embedder_model
        self._sparse_embedder: Any | None = None
        self._dense_embedder: Any | None = None

    def warm_up(self) -> None:
        """Initialize and warm up the embedding models."""

        self._sparse_embedder = FastembedSparseTextEmbedder(
            model=self.sparse_embedder_model,
        )
        self._sparse_embedder.warm_up()

        self._dense_embedder = FastembedTextEmbedder(
            model=self.dense_embedder_model,
        )
        self._dense_embedder.warm_up()

    @component.output_types(documents=list[HaystackDocument])
    def run(
        self, queries: str | list[str], top_k: int | None = None
    ) -> dict[str, list[HaystackDocument]]:
        """
        Run hybrid search with multiple queries.

        Args:
            queries: Single query string or list of query strings
            top_k: Maximum documents to return

        Returns:
            Dict with deduplicated documents sorted by score
        """
        if self._sparse_embedder is None:
            self.warm_up()

        if self._sparse_embedder is None or self._dense_embedder is None:
            raise ValueError("Embedders not initialized after warm_up()")

        # Convert single query to list
        if isinstance(queries, str):
            queries = [queries]

        k = top_k if top_k is not None else self.top_k

        # Run hybrid search for all queries
        results: list[dict[str, list[HaystackDocument]]] = []
        for query in queries:
            sparse_result = self._sparse_embedder.run(text=query)
            dense_result = self._dense_embedder.run(text=query)
            retriever = QdrantHybridRetriever(
                document_store=self.document_store,
                score_threshold=self.min_score,
            )
            result = retriever.run(
                query_sparse_embedding=sparse_result["sparse_embedding"],
                query_embedding=dense_result["embedding"],
                top_k=k,
            )
            results.append(result)

        # Deduplicate and rank using mixin
        docs = self.deduplicate_and_rank(results, k, self.min_score)
        return {"documents": docs}
