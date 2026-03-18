from typing import TYPE_CHECKING, Any

from haystack.tools import ComponentTool

from django_ai_sdk.logger import get_logger
from django_ai_sdk.rags.provider import BaseRAGProvider

if TYPE_CHECKING:
    from django_ai_sdk.assistant import Assistant

logger = get_logger(__name__)


class HaystackRAGProvider(BaseRAGProvider):
    """
    RAG provider for Haystack pipelines.

    Handles warming up Haystack RAG implementations (building indexes),
    caching RAG instances, and converting them to tools on demand.

    Usage:
        class MyAssistant(Assistant):
            rag_provider = HaystackRAGProvider()

            async def get_rag_pipeline(self, silo_id=None):
                # Return a HaystackRAGBase instance
                return QdrantBM25HybridRAG(documents=docs)
    """

    def __init__(self) -> None:
        # Cache: key = "{class_name}_{silo_id}", value = RAG instance
        self._cache: dict[str, Any] = {}

    async def warmup(self, assistant: "Assistant", silo_id: str | None = None) -> None:
        """
        Warm up the Haystack RAG by building indexes.

        Gets or creates the RAG instance and caches the RAG.
        """
        cache_key = self._get_cache_key(assistant, silo_id)
        logger.info(f"Warming up Haystack RAG for {cache_key}")

        # Get or create the RAG instance
        await self._get_or_create_rag(assistant, silo_id)
        logger.info(f"Haystack RAG warmed up for {cache_key}")

    async def get_rag_instance(self, assistant: "Assistant", silo_id: str | None = None) -> Any:
        """
        Get the Haystack RAG instance.

        Returns the cached RAG instance which can be used with:
        - rag.as_tool() to get ComponentTool
        - rag.get_tool(spec) to get ComponentTool with custom spec
        - rag.build_pipeline() for direct pipeline access

        Args:
            assistant: The assistant instance
            silo_id: Optional silo ID for document source

        Returns:
            HaystackRAGBase instance (e.g., QdrantBM25HybridRAG), or None
        """
        return await self._get_or_create_rag(assistant, silo_id)

    async def build_tool(self, rag_instance: Any) -> Any:
        """
        Build a ComponentTool from the Haystack RAG instance.

        Delegates to the RAG's as_tool() method.

        Args:
            rag_instance: The RAG instance from get_rag_instance()

        Returns:
            ComponentTool ready for Haystack ToolAgent, or None
        """
        if rag_instance is None:
            return None

        if hasattr(rag_instance, "as_tool"):
            return rag_instance.as_tool()

        logger.warning("Cannot build ComponentTool: rag_instance does not have as_tool() method")
        return None

    def clear_cache(self) -> None:
        """Clear the RAG cache."""
        self._cache.clear()
        logger.debug("Haystack RAG cache cleared")

    async def reindex(self, assistant: "Assistant", silo_id: str | None = None) -> Any:
        """
        Reindex the RAG by clearing cache and rebuilding.

        Returns the reindexed RAG instance.
        """
        cache_key = self._get_cache_key(assistant, silo_id)
        logger.info(f"Reindexing Haystack RAG for {cache_key}")

        # Clear this entry from cache
        if cache_key in self._cache:
            del self._cache[cache_key]

        # Warm up again (rebuilds indexes and caches)
        await self.warmup(assistant, silo_id)

        # Return the cached RAG
        result = self._cache.get(cache_key)
        logger.info(f"Haystack RAG reindexed for {cache_key}")
        return result

    # TODO: thightly coupled to silo, maybe we want to have some RagKey object
    # that would support many key types (assistant, silo, etc)
    def _get_cache_key(self, assistant: "Assistant", silo_id: str | None) -> str:
        """Generate cache key for this assistant and silo."""
        return f"{assistant.__class__.__name__}_{silo_id or 'default'}"

    async def _get_or_create_rag(
        self,
        assistant: "Assistant",
        silo_id: str | None,
    ) -> Any:
        """
        Get cached RAG or create and cache it.

        Always caches the RAG instance.
        """
        cache_key = self._get_cache_key(assistant, silo_id)

        if cache_key not in self._cache:
            logger.debug(f"Creating Haystack RAG for {cache_key}")

            # Get RAG from assistant
            rag = await assistant.get_rag_pipeline(silo_id)

            if rag is not None:
                # Warm up the RAG
                if hasattr(rag, "warmup") and hasattr(rag, "needs_warmup"):
                    if rag.needs_warmup:
                        logger.debug(f"Warming up RAG for {cache_key}")
                        rag.warmup()

                # Cache the RAG
                self._cache[cache_key] = rag
            else:
                self._cache[cache_key] = None

            logger.debug(f"Haystack RAG created and cached for {cache_key}")
        else:
            logger.debug(f"Using cached Haystack RAG for {cache_key}")

        return self._cache[cache_key]
