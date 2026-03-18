from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from django_ai_sdk.logger import get_logger

if TYPE_CHECKING:
    from django_ai_sdk.assistant import Assistant

logger = get_logger(__name__)


class BaseRAGProvider(ABC):
    """
    Abstract base class for RAG providers.

    RAG providers handle:
    - Warming up RAG (building indexes, loading embeddings)
    - Getting RAG instances for document retrieval
    - Converting RAG instances to tools (framework-specific)

    This abstraction allows the Assistant class to be framework-agnostic.
    Different AI frameworks (Haystack, OpenAI, LangChain) provide their own
    RAGProvider implementations.

    Usage:
        class MyAssistant(Assistant):
            rag_provider = HaystackRAGProvider()

    Or dynamically:
        assistant = MyAssistant()
        assistant.rag_provider = HaystackRAGProvider()
    """

    @abstractmethod
    async def warmup(self, assistant: "Assistant", silo_id: str | None = None) -> None:
        """
        Warm up the RAG by building indexes and loading embeddings.

        This is an expensive operation that should be called before the first
        request to pre-load data structures. The provider should cache the
        warmed-up RAG for subsequent calls.

        Args:
            assistant: The assistant instance to warm up RAG for
            silo_id: Optional silo ID for document source
        """

    @abstractmethod
    async def get_rag_instance(self, assistant: "Assistant", silo_id: str | None = None) -> Any:
        """
        Get the RAG instance for retrieval.

        Returns a framework-specific RAG instance that can be used for
        document retrieval. For Haystack, this might be a RAG adapter
        with a retrieve() method. For OpenAI, this might be a custom
        retrieval implementation.

        Args:
            assistant: The assistant instance
            silo_id: Optional silo ID for document source

        Returns:
            Framework-specific RAG instance, or None if no documents
        """

    @abstractmethod
    async def build_tool(self, rag_instance: Any) -> Any:
        """
        Build a tool from the RAG instance (framework-specific).

        For Haystack, this might wrap the RAG in a ComponentTool.
        For OpenAI, this might create a function tool for retrieval.
        For other frameworks, this might be a no-op if RAG is handled differently.

        Args:
            rag_instance: The RAG instance from get_rag_instance()

        Returns:
            Tool object ready for use in the framework's pipeline/agent, or None
        """

    @abstractmethod
    def clear_cache(self) -> None:
        """Clear the provider's internal cache."""

    @abstractmethod
    async def reindex(self, assistant: "Assistant", silo_id: str | None = None) -> Any:
        """
        Reindex the RAG by clearing cache and rebuilding.

        Args:
            assistant: The assistant instance
            silo_id: Optional silo ID for document source

        Returns:
            The reindexed RAG instance
        """


class RAGProvider(BaseRAGProvider):
    """
    RAG provider for custom/direct RAG implementations.

    This provider handles RAG implementations that inherit from BaseRAGAdapter
    and don't use Haystack pipelines. Examples: BM25RAG, custom vector stores, etc.

    Key features:
    - Caches BaseRAGAdapter instances directly (not wrapped in tools)
    - Calls warmup() to build indexes (expensive, cached)
    - build_tool() creates OpenAI-compatible function tools
    - Works with OpenAIAdapter for context injection

    Usage:
        class MyAssistant(Assistant):
            rag_provider = RAGProvider()

            async def get_rag_pipeline(self, silo_id=None):
                documents = await self.get_rag_documents(silo_id)
                return BM25RAG(documents=documents)

    Flow:
        1. get_rag_instance() → Returns cached BM25RAG
        2. rag.warmup() → Builds BM25 index (expensive, cached)
        3. Adapter calls rag.retrieve(query) → Injects context
        4. build_tool() → Creates OpenAI function (optional)

    Comparison with HaystackRAGProvider:
        - RAGProvider: For direct/custom RAG (BM25, etc.)
        - HaystackRAGProvider: For Haystack pipeline RAG (QdrantBM25HybridRAG, etc.)
    """

    def __init__(self) -> None:
        """Initialize provider with empty cache."""
        self._cache: dict[str, Any] = {}
        logger.debug("RAGProvider initialized")

    async def warmup(self, assistant: "Assistant", silo_id: str | None = None) -> None:
        """
        Warm up the RAG by building the search index.

        Calls the RAG's warmup() method and caches the instance.
        This is an expensive operation that should happen before first use.

        Args:
            assistant: The assistant instance to warm up
            silo_id: Optional silo ID for document source
        """
        cache_key = self._get_cache_key(assistant, silo_id)
        logger.info(f"Warming up Base RAG for {cache_key}")

        # Get RAG from assistant
        rag = await assistant.get_rag_pipeline(silo_id)

        if rag is not None:
            # Warm up the RAG (build index)
            if hasattr(rag, "warmup") and hasattr(rag, "needs_warmup"):
                if rag.needs_warmup:
                    logger.debug(f"Building index for {cache_key}")
                    rag.warmup()

            # Cache the warmed-up RAG
            self._cache[cache_key] = rag
            logger.info(f"Base RAG warmed up and cached for {cache_key}")
        else:
            self._cache[cache_key] = None
            logger.warning(f"No RAG available for {cache_key}")

    async def get_rag_instance(self, assistant: "Assistant", silo_id: str | None = None) -> Any:
        """
        Get the RAG instance (cached or newly created).

        Returns the BaseRAGAdapter instance directly (not wrapped as a tool).
        The instance is cached and warmed up if needed.

        Args:
            assistant: The assistant instance
            silo_id: Optional silo ID for document source

        Returns:
            BaseRAGAdapter instance (e.g., BM25RAG), or None
        """
        cache_key = self._get_cache_key(assistant, silo_id)

        if cache_key not in self._cache:
            logger.debug(f"Creating Base RAG for {cache_key}")
            await self.warmup(assistant, silo_id)
        else:
            logger.debug(f"Using cached Base RAG for {cache_key}")

        return self._cache.get(cache_key)

    async def build_tool(self, rag_instance: Any) -> Callable | None:
        """
        Build a tool from the RAG instance using RAG's as_tool() method.

        The RAG instance (e.g., BM25RAG) should have as_tool() and get_tool() methods
        to create OpenAI-compatible function tools.

        Args:
            rag_instance: The RAG instance from get_rag_instance()

        Returns:
            Callable with name/description attributes, or None if no RAG
        """
        if rag_instance is None:
            return None

        # Check if RAG has as_tool method
        if not hasattr(rag_instance, "as_tool"):
            logger.warning("RAG instance does not have as_tool() method")
            return None

        logger.debug("Building tool using RAG's as_tool() method")

        # Use RAG's as_tool() method
        tool = rag_instance.as_tool()
        return tool

    async def reindex(self, assistant: "Assistant", silo_id: str | None = None) -> Any:
        """
        Reindex the RAG by clearing cache and rebuilding.

        Call this when documents have changed and you need to rebuild the index.

        Args:
            assistant: The assistant instance
            silo_id: Optional silo ID for document source

        Returns:
            The reindexed RAG instance
        """
        cache_key = self._get_cache_key(assistant, silo_id)
        logger.info(f"Reindexing Base RAG for {cache_key}")

        # Clear this entry from cache
        if cache_key in self._cache:
            del self._cache[cache_key]
            logger.debug(f"Cleared cache for {cache_key}")

        # Warm up again (rebuilds index and caches)
        await self.warmup(assistant, silo_id)

        # Return the cached RAG
        result = self._cache.get(cache_key)
        logger.info(f"Base RAG reindexed for {cache_key}")
        return result

    def clear_cache(self) -> None:
        """Clear all cached RAG instances."""
        cache_size = len(self._cache)
        self._cache.clear()
        logger.debug(f"Base RAG cache cleared ({cache_size} entries)")

    def _get_cache_key(self, assistant: "Assistant", silo_id: str | None) -> str:
        """Generate cache key for this assistant and silo."""
        return f"{assistant.__class__.__name__}_{silo_id or 'default'}"
