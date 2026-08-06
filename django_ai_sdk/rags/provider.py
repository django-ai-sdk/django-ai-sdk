from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from django_ai_sdk.logger import get_logger

if TYPE_CHECKING:
    from django_ai_sdk.agent import Agent
    from django_ai_sdk.citations import CitationFormatter, CitationRegistry
    from django_ai_sdk.rags.schemas import RagDocument

logger = get_logger(__name__)


class RAGProvider:
    """
    RAG provider for RAG pipelines.

    Handles warming up RAG implementations (building indexes),
    caching RAG instances, and converting them to tools on demand.

    Usage:
        class MyAgent(Agent):
            rag_provider = RAGProvider()

            async def get_rag_pipeline(self, memory_id=None):
                # Return a RAGBase instance
                return QdrantBM25HybridRAG(documents=docs)
    """

    _MAX_CACHE_SIZE: int = 100

    def __init__(self) -> None:
        # LRU cache: key = "{class_name}_{memory_id}", value = RAG instance.
        # OrderedDict.move_to_end() on hit keeps recently-used entries at the back;
        # popitem(last=False) evicts the LRU entry when the cap is reached.
        self._cache: OrderedDict[str, Any] = OrderedDict()
        # Per-key locks prevent concurrent warmups for the same memory_id.
        # Backends like Qdrant local hold an exclusive file lock during warmup;
        # a second concurrent call for the same key would crash rather than wait.
        # Using one asyncio.Lock per key keeps the fast path (warm cache) lock-free.
        self._warmup_locks: dict[str, asyncio.Lock] = {}

    async def warmup(
        self, agent: Agent, memory_id: str | None = None, force_rebuild: bool = False
    ) -> None:
        """
        Warm up the RAG by building indexes.

        Gets or creates the RAG instance and caches the RAG.

        Args:
            agent: The agent instance
            memory_id: Optional memory ID for document source
            force_rebuild: If True, forces a complete rebuild of the index
        """
        cache_key = self._get_cache_key(agent, memory_id)
        logger.info(f"Warming up RAG for {cache_key} (force_rebuild={force_rebuild})")

        # Get or create the RAG instance
        await self._get_or_create_rag(agent, memory_id, force_rebuild)
        logger.info(f"RAG warmed up for {cache_key}")

    async def get_rag_instance(self, agent: Agent, memory_id: str | None = None) -> Any:
        """
        Get the RAG instance.

        Returns the cached RAG instance which can be used with:
        - rag.as_tool() to get ComponentTool
        - rag.get_tool(spec) to get ComponentTool with custom spec
        - rag.build_pipeline() for direct pipeline access

        Args:
            agent: The agent instance
            memory_id: Optional memory ID for document source

        Returns:
            RAGBase instance (e.g., QdrantBM25HybridRAG), or None
        """
        return await self._get_or_create_rag(agent, memory_id, False)

    def get_cached_rag_instance(self, agent: Agent, memory_id: str | None = None) -> Any:
        """Return the cached RAG instance without warming up or creating a new one.

        Use instead of get_rag_instance() when a second connection would conflict
        (e.g. Qdrant's exclusive local file lock). Returns None if not yet warmed.
        """
        cache_key = self._get_cache_key(agent, memory_id)
        return self._cache.get(cache_key)

    async def get_tool(
        self,
        agent: Agent,
        memory_id: str | None = None,
        *,
        spec: Any = None,
        citation_registry: CitationRegistry | None = None,
        citation_formatter: CitationFormatter | None = None,
    ) -> Any:
        """Get a ready-to-use ComponentTool for the given memory, with optional citations."""
        rag_instance = await self.get_rag_instance(agent, memory_id)
        tool = await self.build_tool(rag_instance, spec=spec)
        if tool is not None and citation_registry is not None and citation_formatter is not None:
            self._attach_citations(tool, citation_formatter, citation_registry)
        return tool

    async def build_tool(self, rag_instance: Any, *, spec: Any = None) -> Any:
        """Build a ComponentTool from a RAG instance.

        Uses the RAG's spec-aware get_tool when a spec is given, otherwise
        falls back to as_tool.

        Args:
            rag_instance: The RAG instance from get_rag_instance()
            spec: Optional tool spec for custom names/descriptions

        Returns:
            ComponentTool ready for ToolAgent, or None
        """
        if rag_instance is None:
            return None
        if spec is not None and hasattr(rag_instance, "get_tool"):
            return await rag_instance.get_tool(spec)
        if hasattr(rag_instance, "as_tool"):
            return await rag_instance.as_tool()
        logger.warning(
            "Cannot build ComponentTool: rag_instance has neither get_tool() nor as_tool()"
        )
        return None

    def _attach_citations(
        self,
        tool: Any,
        formatter: CitationFormatter,
        registry: CitationRegistry,
    ) -> None:
        """Wire a ComponentTool via the citation bridge."""
        from django_ai_sdk.citations.utils import attach_citations  # noqa: PLC0415

        attach_citations(tool, formatter, registry)

    def clear_cache(self) -> None:
        """Clear the RAG cache."""
        self._cache.clear()
        logger.debug("RAG cache cleared")

    async def add_documents(
        self, agent: Agent, memory_id: str | None, documents: list[RagDocument]
    ) -> None:
        """Add documents to existing RAG instance."""
        cache_key = self._get_cache_key(agent, memory_id)
        rag = self._cache.get(cache_key)

        if rag is not None and hasattr(rag, "add_documents"):
            await rag.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to {cache_key}")

    async def remove_documents(
        self, agent: Agent, memory_id: str | None, document_ids: list[str]
    ) -> None:
        """Remove documents from existing RAG instance."""
        cache_key = self._get_cache_key(agent, memory_id)
        rag = self._cache.get(cache_key)

        if rag is not None and hasattr(rag, "remove_documents"):
            await rag.remove_documents(document_ids)
            logger.info(f"Removed {len(document_ids)} documents from {cache_key}")

    async def reindex(
        self, agent: Agent, memory_id: str | None = None, force_rebuild: bool = False
    ) -> Any:
        """
        Reindex the RAG by clearing cache and rebuilding.

        Args:
            agent: The agent instance
            memory_id: Optional memory ID for document source
            force_rebuild: If True, forces a complete rebuild of the index

        Returns:
            The reindexed RAG instance.
        """
        cache_key = self._get_cache_key(agent, memory_id)
        logger.info(f"Reindexing RAG for {cache_key} (force_rebuild={force_rebuild})")

        # Clear this entry from cache
        if cache_key in self._cache:
            del self._cache[cache_key]

        # Warm up again (rebuilds indexes and caches)
        await self.warmup(agent, memory_id, force_rebuild)

        # Return the cached RAG
        result = self._cache.get(cache_key)
        logger.info(f"RAG reindexed for {cache_key}")
        return result

    # TODO: maybe we want to have some RagKey object
    # that would support many key types (agent, memory, etc)
    def _get_cache_key(self, agent: Agent, memory_id: str | None) -> str:
        """Generate cache key for this agent and memory."""
        return f"{agent.__class__.__name__}_{memory_id or 'default'}"

    async def _get_or_create_rag(
        self,
        agent: Agent,
        memory_id: str | None,
        force_rebuild: bool = False,
    ) -> Any:
        """
        Get cached RAG or create and cache it.

        Uses double-checked locking to handle concurrent calls for the same
        cache key safely:
        - Fast path: warm cache is returned immediately without acquiring any lock.
        - Slow path: a per-key asyncio.Lock serializes concurrent warmups so that
          only one coroutine builds the index while others wait, then get the cached
          result instead of repeating the expensive (and potentially exclusive-lock-
          holding) warmup.

        Args:
            agent: The agent instance
            memory_id: Optional memory ID for document source
            force_rebuild: If True, forces a complete rebuild of the index
        """
        cache_key = self._get_cache_key(agent, memory_id)

        # Fast path: return immediately if already cached and no rebuild requested.
        if cache_key in self._cache and not force_rebuild:
            logger.debug(f"Using cached RAG for {cache_key}")
            self._cache.move_to_end(cache_key)  # mark as recently used
            return self._cache[cache_key]

        # Slow path: serialize warmup per key.
        # setdefault is a single dict operation — no await between check and insert,
        # so concurrent coroutines always resolve to the same Lock object for a key.
        lock = self._warmup_locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            # Re-check after acquiring the lock: a previous waiter may have already
            # populated the cache while we were waiting.
            if cache_key in self._cache and not force_rebuild:
                logger.debug(f"Using cached RAG for {cache_key} (post-lock cache hit)")
                self._cache.move_to_end(cache_key)  # mark as recently used
                return self._cache[cache_key]

            logger.debug(f"Creating RAG for {cache_key} (force_rebuild={force_rebuild})")
            rag = await agent.get_rag_pipeline(memory_id)

            if rag is not None:
                if hasattr(rag, "warmup") and hasattr(rag, "needs_warmup"):
                    if rag.needs_warmup or force_rebuild:
                        logger.debug(
                            f"Warming up RAG for {cache_key} (force_rebuild={force_rebuild})"
                        )
                        await rag.warmup(force_rebuild)
                self._cache[cache_key] = rag
            else:
                self._cache[cache_key] = None

            # Evict LRU entries when over the cap; clean up their locks too
            while len(self._cache) > self._MAX_CACHE_SIZE:
                evicted_key, _ = self._cache.popitem(last=False)
                self._warmup_locks.pop(evicted_key, None)
                logger.debug(f"Evicted LRU RAG cache entry: {evicted_key}")

            logger.debug(f"RAG created and cached for {cache_key}")
            return self._cache[cache_key]
