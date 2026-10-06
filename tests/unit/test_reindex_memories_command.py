"""`reindex_memories` rebuilds each memory's index once, from scratch.

The provider is the real RAGProvider; the RAG is a small fake (as in the shared-index
tests), so no Qdrant server or embedding model is needed.
"""

from __future__ import annotations

from io import StringIO
from unittest.mock import AsyncMock, patch

import pytest
from django.core.management import CommandError, call_command

from django_ai_sdk.memories.models import Memory
from django_ai_sdk.rags.provider import RAGProvider


class FakeRag:
    needs_warmup = True

    def __init__(self, memory_id: str) -> None:
        self.memory_id = memory_id
        self.rebuilds: list[bool] = []

    async def warmup(self, force_rebuild: bool = False) -> None:
        self.rebuilds.append(force_rebuild)


class FakeAgent:
    """A RAG agent: a fresh RAG per memory, none for an empty one."""

    rag_provider = RAGProvider()

    def __init__(self, empty: set[str] = frozenset()) -> None:
        self.empty = empty
        self.rags: dict[str, FakeRag] = {}

    async def get_rag_pipeline(self, memory_id: str | None = None) -> FakeRag | None:
        if memory_id in self.empty:
            return None
        return self.rags.setdefault(memory_id, FakeRag(memory_id))


def run(agent: FakeAgent, *args: str) -> str:
    out = StringIO()
    with patch(
        "django_ai_sdk.agents.services.AgentService.get_rag_agents",
        AsyncMock(return_value=[agent]),
    ):
        call_command("reindex_memories", *args, stdout=out)
    return out.getvalue()


@pytest.mark.django_db(transaction=True)
class TestReindexMemories:
    def test_every_memory_is_rebuilt_once_from_scratch(self):
        a, b = Memory.objects.create(name="Test Me"), Memory.objects.create(name="HR")
        empty = Memory.objects.create(name="Empty")
        agent = FakeAgent(empty={str(empty.id)})

        out = run(agent)

        assert {mid: rag.rebuilds for mid, rag in agent.rags.items()} == {
            str(a.id): [True],
            str(b.id): [True],
        }
        assert "Empty" in out and "skipped" in out and "Reindex complete" in out

    def test_says_where_dense_embeddings_are_made(self, settings):
        Memory.objects.create(name="Test Me")

        assert "Dense embeddings: local FastEmbed" in run(FakeAgent())

        settings.AI_SDK_EMBEDDINGS_MODEL = "Qwen/Qwen3-Embedding-8B"
        settings.AI_SDK_EMBEDDINGS_DIM = 4096
        settings.OPENAI_API_URL = "https://api.inference.nebul.io/v1"
        assert (
            "Dense embeddings: remote, Qwen/Qwen3-Embedding-8B (4096 dims) via "
            "https://api.inference.nebul.io/v1"
        ) in run(FakeAgent())

    def test_one_memory(self):
        a = Memory.objects.create(name="Test Me")
        Memory.objects.create(name="HR")
        agent = FakeAgent()

        run(agent, "--memory", str(a.id))

        assert list(agent.rags) == [str(a.id)]

    def test_an_unknown_memory_is_an_error(self):
        with pytest.raises(CommandError, match="not found"):
            run(FakeAgent(), "--memory", "00000000-0000-0000-0000-000000000000")
