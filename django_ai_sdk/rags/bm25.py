"""
BM25-based RAG implementation using bm25s library.

Lightweight keyword search without complex pipelines.
Perfect for OpenAI integration and simple use cases.

This is a direct/custom RAG implementation that inherits from BaseRAGAdapter.
"""

from typing import Any

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
    - Compatible with OpenAIAdapter context injection
    - Can be exposed as OpenAI function tool via BaseRAGProvider

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

    # Explicitly type the config field for BM25Config
    config: BM25Config
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
        # Ensure we have a valid BM25Config
        effective_config = config or BM25Config()
        super().__init__(config=effective_config)
        # Cast config to BM25Config for proper typing
        self.config: BM25Config = effective_config  # type: ignore[assignment]
        # Ensure all documents are RagDocument instances
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

        Implementation details:
        - Uses bm25s.tokenize() for corpus tokenization
        - Creates bm25s.BM25() instance with k1, b parameters
        - Calls index() to build the searchable structure
        - Sets _is_warmed_up = True when complete
        """
        if self._is_warmed_up:
            logger.debug("BM25RAG already warmed up, skipping")
            return

        logger.debug(f"Warming up BM25RAG with {len(self.documents)} documents")

        import bm25s

        # Extract content from RagDocument objects
        corpus = [doc.content for doc in self.documents]

        # Tokenize and build index
        self._corpus_tokens = bm25s.tokenize(corpus)
        bm25_instance = bm25s.BM25(k1=self.config.k1, b=self.config.b)
        bm25_instance.index(self._corpus_tokens)
        self._bm25 = bm25_instance

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

        Example:
            result = await rag.retrieve("pirate treasure")
            # result.documents = [{"content": "...", "score": 0.95}, ...]
            # result.context = formatted string for LLM
            # result.sources = [RAGSource, ...]
        """
        if self.needs_warmup:
            self.warmup()

        if not self._bm25:
            raise RuntimeError("BM25 not initialized. Call warmup() first.")

        import bm25s

        logger.debug(f"Retrieving documents for query: {query[:50]}...")

        # Tokenize query
        query_tokens = bm25s.tokenize([query])

        # Retrieve top-k documents
        results, scores = self._bm25.retrieve(query_tokens, k=self.config.top_k)

        # Build RAGResult
        documents = []
        sources = []

        for i, idx in enumerate(results[0]):
            idx = int(idx)  # Convert from numpy array if needed
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
            context="",  # Will be formatted by format_context
            sources=sources,
            query=query,
        )

        # Format context
        result.context = self.format_context(result)

        return result

    def format_context(self, result: RAGResult) -> str:
        """
        Format retrieval results as context string.

        Includes BM25 scores for relevance ranking visibility.

        Args:
            result: RAGResult from retrieve()

        Returns:
            Formatted context string suitable for LLM injection
        """
        if not result.documents:
            return ""

        context_parts = ["Relevant information from knowledge base:\n"]

        for i, doc in enumerate(result.documents, 1):
            content = doc.get("content", "")
            score = doc.get("score", 0)

            context_parts.append(f"\n[{i}] (relevance: {score:.2f})")
            context_parts.append(f"{content}")

        return "\n".join(context_parts)
