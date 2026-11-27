import asyncio
from unittest.mock import AsyncMock, MagicMock

from django_ai_sdk.rags.provider import RAGProvider
from django_ai_sdk.rags.schemas import RagDocument


def _make_agent(call_counter: list, rag: object) -> MagicMock:
    agent = MagicMock()
    agent.__class__.__name__ = "MockAgent"

    async def get_rag_pipeline(memory_id=None):
        call_counter.append(1)
        await asyncio.sleep(0)
        return rag

    agent.get_rag_pipeline = get_rag_pipeline
    return agent


def _make_rag() -> MagicMock:
    rag = MagicMock()
    rag.needs_warmup = True
    rag.warmup = AsyncMock()
    return rag


class TestRAGProviderConcurrency:
    async def test_concurrent_cold_cache_warms_up_once(self):
        provider = RAGProvider()
        calls: list = []
        rag = _make_rag()
        agent = _make_agent(calls, rag)

        r1, r2 = await asyncio.gather(
            provider.get_rag_instance(agent, "mem-1"),
            provider.get_rag_instance(agent, "mem-1"),
        )

        assert r1 is r2 is rag
        assert len(calls) == 1
        rag.warmup.assert_called_once()

    async def test_warm_cache_never_calls_warmup_again(self):
        provider = RAGProvider()
        calls: list = []
        rag = _make_rag()
        agent = _make_agent(calls, rag)

        await provider.get_rag_instance(agent, "mem-warm")
        await provider.get_rag_instance(agent, "mem-warm")

        assert len(calls) == 1

    async def test_different_keys_warm_up_independently(self):
        provider = RAGProvider()
        calls_a: list = []
        calls_b: list = []
        rag_a, rag_b = _make_rag(), _make_rag()
        agent_a = _make_agent(calls_a, rag_a)
        agent_b = _make_agent(calls_b, rag_b)

        r_a, r_b = await asyncio.gather(
            provider.get_rag_instance(agent_a, "mem-a"),
            provider.get_rag_instance(agent_b, "mem-b"),
        )

        assert r_a is rag_a
        assert r_b is rag_b
        assert len(calls_a) == 1
        assert len(calls_b) == 1
