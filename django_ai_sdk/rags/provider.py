from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from django_ai_sdk.errors import AiSdkError, ErrorCode
from django_ai_sdk.logger import get_logger

if TYPE_CHECKING:
    from django_ai_sdk.adapters.citations import SourceFormatter, SourceRegistry
    from django_ai_sdk.agents.base import Agent
    from django_ai_sdk.rags.schemas import RagDocument

logger = get_logger(__name__)


def _knowledge_unavailable(cache_key: str, exc: Exception) -> AiSdkError:
    if isinstance(exc, AiSdkError):
        return exc
    return AiSdkError(f"RAG {cache_key} failed: {exc}", ErrorCode.KNOWLEDGE_UNAVAILABLE)


class RAGProvider:
    _MAX_CACHE_SIZE: int = 100

    def __init__(self) -> None:
        self._cache: OrderedDict[str, Any] = OrderedDict()
        # Backends like Qdrant local hold an exclusive file lock during warmup;
        # a second concurrent call for the same key would crash rather than wait.
        self._warmup_locks: dict[str, asyncio.Lock] = {}

    async def warmup(
        self, agent: Agent, memory_id: str | None = None, force_rebuild: bool = False
    ) -> None:
        cache_key = self._get_cache_key(agent, memory_id)
        logger.info(f"Warming up RAG for {cache_key} (force_rebuild={force_rebuild})")

        await self._get_or_create_rag(agent, memory_id, force_rebuild)
        logger.info(f"RAG warmed up for {cache_key}")

    async def get_rag_instance(self, agent: Agent, memory_id: str | None = None) -> Any:
        """
        Returns:
            RAGBase instance, or None
        """
        return await self._get_or_create_rag(agent, memory_id, False)

    def get_cached_rag_instance(self, agent: Agent, memory_id: str | None = None) -> Any:
        """Return the cached RAG instance without warming up or creating a new one.
        Returns None if not yet warmed.
        """
        cache_key = self._get_cache_key(agent, memory_id)
        return self._cache.get(cache_key)

    async def get_tool(
        self,
        agent: Agent,
        memory_id: str | None = None,
        *,
        spec: Any = None,
        citation_registry: SourceRegistry | None = None,
        citation_formatter: SourceFormatter | None = None,
    ) -> Any:
        from django_ai_sdk.adapters.citations import collect_sources  # noqa: PLC0415

        rag_instance = await self.get_rag_instance(agent, memory_id)
        tool = await self.build_tool(rag_instance, spec=spec)

        if tool is not None and citation_registry is not None:
            collect_sources(tool, citation_registry, formatter=citation_formatter)
        return tool

    async def build_tool(self, rag_instance: Any, *, spec: Any = None) -> Any:
        """
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

    def clear_cache(self) -> None:
        self._cache.clear()
        logger.debug("RAG cache cleared")

    async def add_documents(
        self, agent: Agent, memory_id: str | None, documents: list[RagDocument]
    ) -> bool:
        """Add documents to this memory's index; True when something was written."""
        cache_key = self._get_cache_key(agent, memory_id)
        try:
            rag = self._cache.get(cache_key) or await self._shared_rag(agent, memory_id)
            if rag is None or not hasattr(rag, "add_documents"):
                return False
            await rag.add_documents(documents)
        except Exception as exc:
            raise _knowledge_unavailable(cache_key, exc) from exc
        logger.info(f"Added {len(documents)} documents to {cache_key}")
        return True

    async def remove_documents(
        self, agent: Agent, memory_id: str | None, document_ids: list[str]
    ) -> bool:
        """Remove documents from this memory's index; True when something was removed."""
        cache_key = self._get_cache_key(agent, memory_id)
        try:
            rag = self._cache.get(cache_key) or await self._shared_rag(agent, memory_id)
            if rag is None or not hasattr(rag, "remove_documents"):
                return False
            await rag.remove_documents(document_ids)
        except Exception as exc:
            raise _knowledge_unavailable(cache_key, exc) from exc
        logger.info(f"Removed {len(document_ids)} documents from {cache_key}")
        return True

    async def _shared_rag(self, agent: Agent, memory_id: str | None) -> Any:
        rag = await agent.get_rag_pipeline(memory_id)
        storage = getattr(getattr(rag, "config", None), "storage", None)
        if not getattr(storage, "is_server", False):
            return None
        await rag.warmup()
        return rag

    async def reindex(
        self, agent: Agent, memory_id: str | None = None, force_rebuild: bool = False
    ) -> Any:
        """Reindex the RAG by clearing cache and rebuilding."""
        cache_key = self._get_cache_key(agent, memory_id)
        logger.info(f"Reindexing RAG for {cache_key} (force_rebuild={force_rebuild})")

        if cache_key in self._cache:
            del self._cache[cache_key]

        await self.warmup(agent, memory_id, force_rebuild)

        result = self._cache.get(cache_key)
        logger.info(f"RAG reindexed for {cache_key}")
        return result

    def _get_cache_key(self, agent: Agent, memory_id: str | None) -> str:
        return f"{agent.__class__.__name__}_{memory_id or 'default'}"

    async def _get_or_create_rag(
        self,
        agent: Agent,
        memory_id: str | None,
        force_rebuild: bool = False,
    ) -> Any:
        cache_key = self._get_cache_key(agent, memory_id)

        if cache_key in self._cache and not force_rebuild:
            logger.debug(f"Using cached RAG for {cache_key}")
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        lock = self._warmup_locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            if cache_key in self._cache and not force_rebuild:
                logger.debug(f"Using cached RAG for {cache_key} (post-lock cache hit)")
                self._cache.move_to_end(cache_key)
                return self._cache[cache_key]

            logger.debug(f"Creating RAG for {cache_key} (force_rebuild={force_rebuild})")
            try:
                rag = await agent.get_rag_pipeline(memory_id)
                if rag is not None and hasattr(rag, "warmup") and hasattr(rag, "needs_warmup"):
                    if rag.needs_warmup or force_rebuild:
                        logger.debug(
                            f"Warming up RAG for {cache_key} (force_rebuild={force_rebuild})"
                        )
                        await rag.warmup(force_rebuild)
            except Exception as exc:
                raise _knowledge_unavailable(cache_key, exc) from exc
            if rag is None:
                return None
            self._cache[cache_key] = rag

            while len(self._cache) > self._MAX_CACHE_SIZE:
                evicted_key, _ = self._cache.popitem(last=False)
                self._warmup_locks.pop(evicted_key, None)
                logger.debug(f"Evicted LRU RAG cache entry: {evicted_key}")

            logger.debug(f"RAG created and cached for {cache_key}")
            return self._cache[cache_key]
