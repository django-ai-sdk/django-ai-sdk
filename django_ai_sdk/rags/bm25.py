from typing import Any

import bm25s
from pydantic import BaseModel

from django_ai_sdk.logger import get_logger
from django_ai_sdk.rags.base import BaseRAGAdapter, RAGResult, RAGSource
from django_ai_sdk.rags.schemas import RagDocument

logger = get_logger(__name__)


class BM25Config(BaseModel):
    """
    Configuration for BM25RAG.

    Attributes:
        top_k: Number of documents to retrieve (default: 5)
        k1: BM25 k1 parameter (default: 1.5) - term frequency saturation
        b: BM25 b parameter (default: 0.75) - length normalization
    """

    top_k: int = 5
    k1: float = 1.5
    b: float = 0.75


class BM25RAG(BaseRAGAdapter):
    """
    BM25-based RAG implementation using bm25s library.

    This is a lightweight RAG that uses pure BM25 keyword search without
    embeddings or complex pipelines. It inherits from BaseRAGAdapter and
    is managed by BaseRAGProvider.

    Features:
    - Fast keyword-based retrieval
    - No GPU required
    - Minimal dependencies (just bm25s)

    Example:
        from django_ai_sdk.rags import RagDocument

        documents = [
            RagDocument(id="1", content="Python is a programming language..."),
            RagDocument(id="2", content="Django is a web framework...")
        ]

        rag = BM25RAG(
            documents=documents,
            config=BM25Config(top_k=5)
        )

        # Warm up (builds index)
        rag.warmup()

        # Retrieve
        result = await rag.retrieve("What is Python?")
        print(result.context)
    """

    documents: list[RagDocument]
    _bm25: Any | None
    _corpus_tokens: Any | None

    def __init__(
        self,
        documents: list[RagDocument],
        config: BM25Config | None = None,
    ) -> None:
        """
        Initialize BM25 RAG.

        Args:
            documents: List of RagDocument objects
            config: Configuration for retrieval parameters (defaults to BM25Config())
        """
        self.config: BM25Config = config or BM25Config()  # type: ignore[assignment]

        self.documents = [
            doc if isinstance(doc, RagDocument) else RagDocument.from_dict(doc) for doc in documents
        ]
        self._bm25 = None
        self._corpus_tokens = None
        logger.debug(f"BM25RAG initialized with {len(documents)} documents")

    def warmup(self) -> None:
        """
        Warm up by building the BM25 index.
        This tokenizes all documents and builds the BM25 index.
        It's an expensive operation that should be cached by the provider.
        """
        if self._is_warmed_up:
            logger.debug("BM25RAG already warmed up, skipping")
            return

        logger.debug(f"Warming up BM25RAG with {len(self.documents)} documents")

        # Extract content from RagDocument objects
        corpus = [doc.content for doc in self.documents]

        # Tokenize and build index
        self._corpus_tokens = bm25s.tokenize(corpus)
        self._bm25 = bm25s.BM25(k1=self.config.k1, b=self.config.b)
        self._bm25.index(self._corpus_tokens)

        self._is_warmed_up = True
        logger.debug("BM25RAG warmup complete")

    async def retrieve(self, query: str) -> RAGResult:
        """
        Retrieve relevant documents using BM25.

        Automatically calls warmup() if needed (checks needs_warmup property).

        Args:
            query: Search query string

        Returns:
            RAGResult with documents, context, and sources
        """

        # TODO: not sure if this is a good location to warmup
        if self.needs_warmup:
            self.warmup()

        if not self._bm25:
            raise RuntimeError("BM25 not initialized. Call warmup() first.")

        logger.debug(f"Retrieving documents for query: {query[:50]}...")

        # Tokenize query
        query_tokens = bm25s.tokenize([query])

        # Retrieve top-k documents
        results, scores = self._bm25.retrieve(query_tokens, k=self.config.top_k)

        # Build RAGResult
        documents = []
        sources = []

        for i, idx in enumerate(results[0]):
            idx = int(idx)
            doc = self.documents[idx]
            score = float(scores[0][i])

            # Set score on document
            doc.score = score

            documents.append(doc.to_dict())
            sources.append(RAGSource(id=doc.id, content=doc.content, metadata=doc.metadata))

        logger.debug(f"Retrieved {len(documents)} documents")

        # Create RAGResult
        result = RAGResult(
            documents=documents,
            context="",
            sources=sources,
            query=query,
        )

        # Format context
        result.context = self.format_context(result)

        return result

    # TODO: add self.config.context_prompt
    # And we might want to pass this function from config as well
    # That way SDK users can override the formatting logic
    def format_context(self, result: RAGResult) -> str:
        """
        Format retrieval results as context string.

        Includes BM25 scores for relevance ranking visibility.

        Args:
            result: RAGResult from retrieve()

        Returns:
            Formatted context string suitable for LLM injection
        """

        # TODO: we probably want to raise some Exception, like RetrievalError
        # Because we might want stream different messages from the adapter.
        if not result.documents:
            return ""

        parts = ["Relevant information from knowledge base:\n"]

        for i, doc in enumerate(result.documents, 1):
            content = doc.get("content", "")
            score = doc.get("score", 0)

            parts.append(f"\n[{i}] (relevance: {score:.2f})")
            parts.append(f"{content}")

        return "\n".join(parts)
