from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import bm25s
from django_ai_sdk.logger import get_logger
from django_ai_sdk.rags.base import BaseRAGAdapter, RAGResult, RAGSource
from django_ai_sdk.rags.schemas import RagDocument
from pydantic import BaseModel

if TYPE_CHECKING:
    from django_ai_sdk.rags.schemas import ToolSpec

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
    _doc_id_to_index: dict[str, int]

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
        self.config: BM25Config = config or BM25Config()

        self.documents = [
            doc if isinstance(doc, RagDocument) else RagDocument.from_dict(doc) for doc in documents
        ]
        self._bm25 = None
        self._corpus_tokens = None
        self._doc_id_to_index = {}
        logger.debug(f"BM25RAG initialized with {len(documents)} documents")

    def _rebuild_index(self) -> None:
        """Rebuild the BM25 index from self.documents."""
        if not self.documents:
            # No documents to index
            self._bm25 = None
            self._corpus_tokens = None
            self._doc_id_to_index = {}
            logger.debug("No documents to index, skipping BM25 index build")
            return

        corpus = [doc.content for doc in self.documents]
        self._doc_id_to_index = {doc.id: i for i, doc in enumerate(self.documents)}

        self._corpus_tokens = bm25s.tokenize(corpus)
        self._bm25 = bm25s.BM25(k1=self.config.k1, b=self.config.b)
        self._bm25.index(self._corpus_tokens)

        logger.debug(f"BM25 index rebuilt with {len(self.documents)} documents")

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
        self._rebuild_index()
        self._is_warmed_up = True
        logger.debug("BM25RAG warmup complete")

    async def add_documents(self, documents: list[RagDocument]) -> None:
        """
        Add documents to the BM25 index incrementally.

        Since bm25s doesn't support incremental adds, we rebuild the index
        with all documents (existing + new).

        Args:
            documents: List of RagDocument objects to add.
        """
        logger.info(f"Adding {len(documents)} documents to BM25RAG")
        self.documents.extend(documents)
        self._rebuild_index()
        logger.info(f"BM25RAG now has {len(self.documents)} documents")

    async def remove_documents(self, document_ids: list[str]) -> None:
        """
        Remove documents from the BM25 index incrementally.

        Since bm25s doesn't support incremental removes, we rebuild the index
        without the removed documents.

        Args:
            document_ids: List of document IDs to remove.
        """
        removed = set(document_ids)
        before_count = len(self.documents)
        self.documents = [d for d in self.documents if d.id not in removed]
        after_count = len(self.documents)

        logger.info(f"Removing {before_count - after_count} documents from BM25RAG")
        self._rebuild_index()
        logger.info(f"BM25RAG now has {len(self.documents)} documents")

    def refresh_documents(self, documents: list[RagDocument]) -> None:
        """
        Fully refresh the index with a new set of documents.

        Replaces all documents and rebuilds the index.

        Args:
            documents: The new complete set of documents.
        """
        logger.info(f"Refreshing BM25RAG with {len(documents)} documents")
        self.documents = [
            doc if isinstance(doc, RagDocument) else RagDocument.from_dict(doc) for doc in documents
        ]
        self._rebuild_index()
        logger.info(f"BM25RAG refresh complete: {len(documents)} documents")

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

        # Retrieve top-k documents (cap at number of documents available)
        k = min(self.config.top_k, len(self.documents))
        if k == 0:
            return RAGResult(documents=[], context="", sources=[], query=query)

        results, scores = self._bm25.retrieve(query_tokens, k=k)

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

    def as_tool(self) -> Callable:
        """
        Return BM25 RAG as a tool callable.

        Returns:
            Callable that accepts query and returns documents.
        """

        async def search(query: str) -> dict:
            result = await self.retrieve(query)
            return {
                "documents": [
                    {"id": d["id"], "content": d["content"], "score": d.get("score", 0)}
                    for d in result.documents
                ]
            }

        # Create a function-like object with metadata
        search.name = "bm25_search"
        search.description = "Search documents using BM25 keyword retrieval"
        return search

    def get_tool(self, spec: "ToolSpec") -> Callable:
        """
        Get tool with custom specification.

        Args:
            spec: ToolSpec with name and description.

        Returns:
            Tool callable with customized name/description.
        """
        tool = self.as_tool()
        tool.name = spec.name
        tool.description = spec.description
        return tool

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
