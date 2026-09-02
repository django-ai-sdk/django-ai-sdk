from __future__ import annotations

import asyncio
from typing import Any

from haystack import component
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import Document as HaystackDocument

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

    @component.output_types(documents=list[HaystackDocument])
    async def run_async(
        self, queries: list[str], top_k: int | None = None
    ) -> dict[str, list[HaystackDocument]]:
        k = top_k if top_k is not None else self.top_k
        results: list[dict[str, list[HaystackDocument]]] = []
        for query in queries:
            result = await asyncio.to_thread(self.retrieve, query, k)
            results.append(result)
        docs = self.deduplicate_and_rank(results, k, self.min_score)
        return {"documents": docs}


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
