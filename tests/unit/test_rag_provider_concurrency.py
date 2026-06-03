"""
Concurrency tests for HaystackRAGProvider and RAGProvider.

Verifies that concurrent cold-cache calls for the same key serialize on
warmup (double-checked locking) rather than racing - which would crash
backends that hold an exclusive resource during warmup (e.g. Qdrant local
file lock).

The `await asyncio.sleep(0)` in the mock get_rag_pipeline is intentional:
it yields control to the event loop so the two coroutines actually interleave
at the point where the race would occur without the lock. Without it the first
call would complete synchronously and the test would pass trivially.
"""

import asyncio
from unittest.mock import MagicMock

from django_ai_sdk.rags import RAGProvider
from django_ai_sdk.rags.haystack.provider import HaystackRAGProvider
from django_ai_sdk.rags.schemas import RagDocument


def _make_assistant(call_counter: list, rag: object) -> MagicMock:
    """Mock assistant that yields once before returning the RAG instance."""
    assistant = MagicMock()
    assistant.__class__.__name__ = "MockAssistant"

    async def get_rag_pipeline(memory_id=None):
        call_counter.append(1)
        await asyncio.sleep(0)  # yield — lets the second coroutine enter before warmup completes
        return rag

    assistant.get_rag_pipeline = get_rag_pipeline
    return assistant


def _make_rag() -> MagicMock:
    rag = MagicMock()
    rag.needs_warmup = True
    rag.warmup = MagicMock()
    return rag


class TestHaystackRAGProviderConcurrency:
    async def test_concurrent_cold_cache_warms_up_once(self):
        """Two simultaneous cold-cache calls must trigger exactly one warmup."""
        provider = HaystackRAGProvider()
        calls: list = []
        rag = _make_rag()
        assistant = _make_assistant(calls, rag)

        r1, r2 = await asyncio.gather(
            provider.get_rag_instance(assistant, "mem-1"),
            provider.get_rag_instance(assistant, "mem-1"),
        )

        assert r1 is r2 is rag
        assert len(calls) == 1
        rag.warmup.assert_called_once()

    async def test_warm_cache_never_calls_warmup_again(self):
        """After the cache is warm, subsequent calls skip warmup entirely."""
        provider = HaystackRAGProvider()
        calls: list = []
        rag = _make_rag()
        assistant = _make_assistant(calls, rag)

        await provider.get_rag_instance(assistant, "mem-warm")
        await provider.get_rag_instance(assistant, "mem-warm")

        assert len(calls) == 1

    async def test_different_keys_warm_up_independently(self):
        """Calls for different keys do not interfere with each other."""
        provider = HaystackRAGProvider()
        calls_a: list = []
        calls_b: list = []
        rag_a, rag_b = _make_rag(), _make_rag()
        assistant_a = _make_assistant(calls_a, rag_a)
        assistant_b = _make_assistant(calls_b, rag_b)

        r_a, r_b = await asyncio.gather(
            provider.get_rag_instance(assistant_a, "mem-a"),
            provider.get_rag_instance(assistant_b, "mem-b"),
        )

        assert r_a is rag_a
        assert r_b is rag_b
        assert len(calls_a) == 1
        assert len(calls_b) == 1


class TestRAGProviderConcurrency:
    async def test_concurrent_cold_cache_warms_up_once(self):
        """RAGProvider has the same guarantee as HaystackRAGProvider."""
        provider = RAGProvider()
        calls: list = []
        rag = _make_rag()
        assistant = _make_assistant(calls, rag)

        r1, r2 = await asyncio.gather(
            provider.get_rag_instance(assistant, "mem-1"),
            provider.get_rag_instance(assistant, "mem-1"),
        )

        assert r1 is r2
        assert len(calls) == 1

    async def test_warm_cache_never_calls_warmup_again(self):
        provider = RAGProvider()
        calls: list = []
        rag = _make_rag()
        assistant = _make_assistant(calls, rag)

        await provider.get_rag_instance(assistant, "mem-warm")
        await provider.get_rag_instance(assistant, "mem-warm")

        assert len(calls) == 1
