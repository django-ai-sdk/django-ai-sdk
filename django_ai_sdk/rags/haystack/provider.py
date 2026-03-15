"""
Haystack-specific RAG provider for the Django AI SDK.

Implements the RAGProvider interface for Haystack pipelines.
"""

from typing import TYPE_CHECKING, Any

from django_ai_sdk.logger import get_logger
from django_ai_sdk.rags.provider import RAGProvider

if TYPE_CHECKING:
    from django_ai_sdk.assistant import Assistant

logger = get_logger(__name__)


class HaystackRAGProvider(RAGProvider):
    """
    RAG provider for Haystack pipelines.

    Handles warming up Haystack RAG implementations (building indexes),
    caching, and converting RAG instances to ComponentTools.

    Usage:
        class MyAssistant(Assistant):
            rag_provider = HaystackRAGProvider()

            async def get_rag_pipeline(self, silo_id=None):
                # Return a HaystackRAGBase instance
                return QdrantBM25HybridRAG(documents=docs)
    """

    def __init__(self) -> None:
        # Cache: key = "{class_name}_{silo_id}", value = RAG instance or ComponentTool
        self._cache: dict[str, Any] = {}

    async def warmup(self, assistant: "Assistant", silo_id: str | None = None) -> None:
        """
        Warm up the Haystack RAG by building indexes.

        Creates an OpenAIChatGenerator and calls the RAG's warmup() method,
        then caches the resulting ComponentTool.
        """
        cache_key = self._get_cache_key(assistant, silo_id)
        logger.info(f"Warming up Haystack RAG for {cache_key}")

        # Import Haystack-specific components
        from haystack.components.generators.chat import OpenAIChatGenerator
        from haystack.utils import Secret

        # Create generator (needed for ComponentTool)
        generator = OpenAIChatGenerator(
            model=assistant.get_model(),
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
        )

        # Get or create the RAG instance with tool
        await self._get_or_create_rag(assistant, silo_id, generator)
        logger.info(f"Haystack RAG warmed up for {cache_key}")

    async def get_rag_instance(self, assistant: "Assistant", silo_id: str | None = None) -> Any:
        """
        Get the Haystack RAG instance.

        Returns the raw RAG adapter (not wrapped as a tool).
        """
        return await self._get_or_create_rag(assistant, silo_id, generator=None)

    async def build_tool(self, rag_instance: Any, generator: Any | None = None) -> Any:
        """
        Build a ComponentTool from the Haystack RAG instance.

        If the rag_instance is already a ComponentTool, returns it directly.
        Otherwise, calls rag_instance.as_tool() to build the tool.

        Note: The generator parameter is accepted for interface compatibility
        but Haystack RAG implementations create ComponentTools internally
        without needing an external generator.
        """
        if rag_instance is None:
            return None

        # Check if already a ComponentTool
        from haystack.tools import ComponentTool

        if isinstance(rag_instance, ComponentTool):
            return rag_instance

        # Build tool - generator is not needed for Haystack as_tool()
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

        Returns the reindexed RAG as a ComponentTool.
        """
        cache_key = self._get_cache_key(assistant, silo_id)
        logger.info(f"Reindexing Haystack RAG for {cache_key}")

        # Clear this entry from cache
        if cache_key in self._cache:
            del self._cache[cache_key]

        # Warm up again (rebuilds indexes and caches)
        await self.warmup(assistant, silo_id)

        # Return the cached tool
        result = self._cache.get(cache_key)
        logger.info(f"Haystack RAG reindexed for {cache_key}")
        return result

    def _get_cache_key(self, assistant: "Assistant", silo_id: str | None) -> str:
        """Generate cache key for this assistant and silo."""
        return f"{assistant.__class__.__name__}_{silo_id or 'default'}"

    async def _get_or_create_rag(
        self,
        assistant: "Assistant",
        silo_id: str | None,
        generator: Any | None = None,
    ) -> Any:
        """
        Get cached RAG or create and cache it.

        If generator is provided, builds and caches ComponentTool.
        If not, caches the raw RAG instance.
        """
        cache_key = self._get_cache_key(assistant, silo_id)

        if cache_key not in self._cache:
            logger.debug(f"Creating Haystack RAG for {cache_key}")

            # Get RAG from assistant
            rag = await assistant.get_rag_pipeline(silo_id)

            if rag is not None:
                # Warm up the RAG (build indexes)
                if hasattr(rag, "warmup") and hasattr(rag, "needs_warmup"):
                    if rag.needs_warmup:
                        logger.debug(f"Warming up RAG for {cache_key}")
                        rag.warmup()

                # Build tool - generator is not needed for Haystack as_tool()
                if hasattr(rag, "as_tool"):
                    logger.debug(f"Building ComponentTool for {cache_key}")
                    self._cache[cache_key] = rag.as_tool()
                else:
                    self._cache[cache_key] = rag
            else:
                self._cache[cache_key] = None

            logger.debug(f"Haystack RAG created and cached for {cache_key}")
        else:
            logger.debug(f"Using cached Haystack RAG for {cache_key}")

            # If we have raw RAG but need tool, build it
            cached = self._cache[cache_key]
            if cached is not None and not self._is_component_tool(cached):
                if hasattr(cached, "as_tool"):
                    logger.debug(f"Building ComponentTool from cached RAG for {cache_key}")
                    self._cache[cache_key] = cached.as_tool()

        return self._cache[cache_key]

    def _is_component_tool(self, obj: Any) -> bool:
        """Check if object is a Haystack ComponentTool."""
        from haystack.tools import ComponentTool

        return isinstance(obj, ComponentTool)
