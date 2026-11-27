from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from typing import Any

from haystack import component
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import Document as HaystackDocument

from django_ai_sdk.logger import get_logger

logger = get_logger(__name__)

CANDIDATES_PER_QUERY = 4
RRF_K = 60


def _words(text: str) -> set[str]:
    text = text.lower()
    return set(re.findall(r"\w+", text)) | set(re.findall(r"\w+(?:[-./]\w+)+", text))


def _name(doc: HaystackDocument) -> str:
    meta = doc.meta or {}
    return str(meta.get("name") or meta.get("file_name") or meta.get("title") or "")


class MultiQueryDeduplicationMixin:
    @staticmethod
    def deduplicate_and_rank(
        results: list[dict[str, list[HaystackDocument]]],
        top_k: int,
        min_score: float | None = None,
    ) -> list[HaystackDocument]:
        """
        Returns:
            List of unique documents sorted by score (descending), limited to top_k
        """
        all_docs: dict[str, HaystackDocument] = {}
        for result in results:
            for doc in result.get("documents", []):
                if doc.id not in all_docs:
                    all_docs[doc.id] = doc

        docs = list(all_docs.values())
        docs.sort(
            key=lambda x: x.score if hasattr(x, "score") and x.score is not None else 0.0,
            reverse=True,
        )

        if min_score is not None:
            docs = [d for d in docs if (d.score or 0.0) >= min_score]

        return docs[:top_k]

    @staticmethod
    def fuse_and_rank(
        results: list[dict[str, list[HaystackDocument]]],
        queries: list[str],
        top_k: int,
        min_score: float | None = None,
        query: str | None = None,
    ) -> list[HaystackDocument]:
        """
        Returns:
            Unique documents, best first, at most top_k (scores unchanged)
        """
        rankings = [
            [
                d
                for d in r.get("documents", [])
                if min_score is None or (d.score or 0.0) >= min_score
            ]
            for r in results
        ]
        found = {doc.id: doc for ranking in rankings for doc in ranking}
        words = {
            doc_id: _words(f"{_name(doc)} {doc.content or ''}") for doc_id, doc in found.items()
        }
        terms = _words(query or (queries[0] if queries else ""))
        terms = {t for t in terms if sum(t in w for w in words.values()) <= len(found) / 2}
        matched = {doc_id: len(terms & w) for doc_id, w in words.items()}
        exact = sorted(
            (doc for doc in found.values() if matched[doc.id]),
            key=lambda doc: matched[doc.id],
            reverse=True,
        )
        fused: defaultdict[str, float] = defaultdict(float)
        for ranking in rankings:
            for rank, doc in enumerate(ranking, 1):
                fused[doc.id] += 1 / (RRF_K + rank)
        most = max(matched.values(), default=0) or 1
        for rank, doc in enumerate(exact, 1):
            fused[doc.id] += len(rankings) * (matched[doc.id] / most) / (RRF_K + rank)
        docs = sorted(found.values(), key=lambda doc: fused[doc.id], reverse=True)[:top_k]
        logger.info(
            "RAG search: {} queries {}, hits per query {}, exact terms {} in {} -> {} kept",
            len(queries),
            queries,
            [len(r) for r in rankings],
            sorted(terms),
            len(exact),
            len(docs),
        )
        return docs


@component
class BaseMultiQueryRetriever(MultiQueryDeduplicationMixin):
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
        """Subclasses must override this method."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement retrieve()")

    @component.output_types(documents=list[HaystackDocument])
    def run(
        self, queries: list[str], top_k: int | None = None, query: str | None = None
    ) -> dict[str, list[HaystackDocument]]:
        """
        Run multiple queries and return deduplicated results.

        Args:
            queries: List of query strings to execute
            top_k: Maximum documents to return (uses self.top_k if None)
            query: The user's query before expansion (its words count verbatim)

        Returns:
            Dict with "documents" key containing unique documents sorted by score
        """
        k = top_k if top_k is not None else self.top_k

        results: list[dict[str, list[HaystackDocument]]] = []
        for q in queries:
            results.append(self.retrieve(q, k * CANDIDATES_PER_QUERY))

        docs = self.fuse_and_rank(results, queries, k, self.min_score, query=query)
        return {"documents": docs}

    @component.output_types(documents=list[HaystackDocument])
    async def run_async(
        self, queries: list[str], top_k: int | None = None, query: str | None = None
    ) -> dict[str, list[HaystackDocument]]:
        k = top_k if top_k is not None else self.top_k
        results: list[dict[str, list[HaystackDocument]]] = []
        for q in queries:
            results.append(await asyncio.to_thread(self.retrieve, q, k * CANDIDATES_PER_QUERY))
        docs = self.fuse_and_rank(results, queries, k, self.min_score, query=query)
        return {"documents": docs}


@component
class MultiQueryBM25Retriever(BaseMultiQueryRetriever):
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
        return self._retriever.run(query=query, top_k=top_k)
