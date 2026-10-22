"""`reindex_memories` rebuilds each out-of-date memory's index once, from scratch.

The provider is the real RAGProvider; the RAG is a small fake (as in the shared-index
tests), so no Qdrant server or embedding model is needed.
"""

from __future__ import annotations

from io import StringIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from django.core.management import CommandError, call_command

from django_ai_sdk.memories.models import Memory
from django_ai_sdk.rags.provider import RAGProvider


class FakeRag:
    needs_warmup = True

    def __init__(
        self, memory_id: str, dim: int = 384, fail: bool = False, missing: int = 0
    ) -> None:
        self.memory_id = memory_id
        self.missing = missing
        self.rebuilds: list[bool] = []
        self.fail = fail
        self.config = SimpleNamespace(
            dense_embedder_model="BAAI/bge-small-en-v1.5",
            sparse_embedder_model="Qdrant/bm42",
            embedding_dim=dim,
            storage=SimpleNamespace(is_server=True, location="http://qdrant:6333"),
        )

    async def warmup(self, force_rebuild: bool = False) -> None:
        if self.fail:
            raise RuntimeError("qdrant down")
        self.rebuilds.append(force_rebuild)

    async def embed_missing(self) -> tuple[int, int]:
        self.embedded = self.missing
        return self.missing, 0


class FakeAgent:
    """A RAG agent: a fresh RAG per memory, none for an empty one."""

    rag_provider = RAGProvider()

    def __init__(
        self, empty: set[str] = frozenset(), dim: int = 384, fail: bool = False, missing: int = 0
    ) -> None:
        self.empty = empty
        self.missing = missing
        self.dim = dim
        self.fail = fail
        self.rags: dict[str, FakeRag] = {}

    async def get_rag_pipeline(self, memory_id: str | None = None) -> FakeRag | None:
        if memory_id in self.empty:
            return None
        return self.rags.setdefault(
            memory_id, FakeRag(memory_id, self.dim, self.fail, self.missing)
        )


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

    def test_says_where_the_index_lives(self):
        Memory.objects.create(name="Test Me")

        assert "Vector store: remote, http://qdrant:6333" in run(FakeAgent())

    def test_stores_what_the_index_was_built_with(self):
        memory = Memory.objects.create(name="Test Me", metadata={"owner": "hr"})

        run(FakeAgent())

        memory.refresh_from_db()
        stored = memory.metadata["sdk"]["rag"]
        assert stored.pop("indexed_at")
        assert stored == {
            "rag": "FakeRag",
            "dense_model": "BAAI/bge-small-en-v1.5",
            "sparse_model": "Qdrant/bm42",
            "dim": 384,
        }
        assert memory.metadata["owner"] == "hr"

    def test_an_up_to_date_memory_is_skipped(self):
        Memory.objects.create(name="Test Me")
        run(FakeAgent())
        agent = FakeAgent()

        out = run(agent)

        assert all(rag.rebuilds == [] for rag in agent.rags.values())
        assert "up to date, skipped" in out

    def test_another_dimension_rebuilds(self):
        memory = Memory.objects.create(name="Test Me")
        run(FakeAgent())
        agent = FakeAgent(dim=4096)

        run(agent)

        assert agent.rags[str(memory.id)].rebuilds == [True]
        memory.refresh_from_db()
        assert memory.metadata["sdk"]["rag"]["dim"] == 4096

    def test_force_rebuilds_up_to_date_memories_and_stores_again(self):
        memory = Memory.objects.create(name="Test Me")
        run(FakeAgent())
        memory.refresh_from_db()
        before = memory.metadata["sdk"]["rag"]["indexed_at"]
        agent = FakeAgent()

        run(agent, "--force")

        assert agent.rags[str(memory.id)].rebuilds == [True]
        memory.refresh_from_db()
        assert memory.metadata["sdk"]["rag"]["indexed_at"] > before

    def test_a_failed_rebuild_stores_nothing(self):
        memory = Memory.objects.create(name="Test Me")

        with pytest.raises(CommandError, match="1 of 1 memories failed"):
            run(FakeAgent(fail=True))

        memory.refresh_from_db()
        assert "sdk" not in memory.metadata

    def test_missing_only_embeds_the_missing_chunks(self):
        memory = Memory.objects.create(name="Test Me")
        run(FakeAgent())
        agent = FakeAgent(missing=3)

        out = run(agent, "--missing")

        rag = agent.rags[str(memory.id)]
        assert (rag.rebuilds, rag.embedded) == ([], 3)
        assert "3 chunks embedded" in out

    def test_missing_skips_an_index_of_another_model_with_a_warning(self):
        memory = Memory.objects.create(name="Test Me")
        run(FakeAgent())
        agent = FakeAgent(dim=4096, missing=3)

        out = run(agent, "--missing")

        rag = agent.rags[str(memory.id)]
        assert rag.rebuilds == [] and not hasattr(rag, "embedded")
        assert "dim=384, not the current model" in out and "Reindex complete" in out

    def test_missing_fills_in_a_memory_without_a_signature_and_adopts_it(self):
        memory = Memory.objects.create(name="Test Me")
        agent = FakeAgent(missing=3)

        out = run(agent, "--missing")

        rag = agent.rags[str(memory.id)]
        assert (rag.rebuilds, rag.embedded) == ([], 3)
        assert "3 chunks embedded (no signature yet" in out
        memory.refresh_from_db()
        assert memory.metadata["sdk"]["rag"]["adopted"] is True

        assert "complete" in run(FakeAgent(), "--missing")  # matches from now on

    def test_missing_fails_on_an_unsigned_index_of_another_dimension(self):
        memory = Memory.objects.create(name="Test Me")
        refused = ValueError("Collection has a different embedding dimension")

        with (
            patch.object(FakeRag, "embed_missing", AsyncMock(side_effect=refused)),
            pytest.raises(CommandError, match="1 of 1 memories failed"),
        ):
            run(FakeAgent(), "--missing")

        memory.refresh_from_db()
        assert "sdk" not in memory.metadata

    def test_missing_fails_when_chunks_stay_without_a_vector(self):
        Memory.objects.create(name="Test Me")
        run(FakeAgent())
        agent = FakeAgent()
        with (
            patch.object(FakeRag, "embed_missing", AsyncMock(return_value=(1, 2))),
            pytest.raises(CommandError, match="1 of 1 memories failed"),
        ):
            run(agent, "--missing")

    def test_force_and_missing_are_exclusive(self):
        with pytest.raises(CommandError, match="not allowed with"):
            run(FakeAgent(), "--force", "--missing")

    def test_without_missing_gaps_are_not_checked(self):
        Memory.objects.create(name="Test Me")
        run(FakeAgent())
        agent = FakeAgent(missing=3)

        assert "up to date, skipped" in run(agent)

    def test_a_rebuild_that_leaves_gaps_fails_and_stores_nothing(self):
        memory = Memory.objects.create(name="Test Me")

        with pytest.raises(CommandError, match="1 of 1 memories failed"):
            with patch.object(FakeRag, "embed_missing", AsyncMock(return_value=(0, 2))):
                run(FakeAgent())

        memory.refresh_from_db()
        assert "sdk" not in memory.metadata

    def test_one_memory(self):
        a = Memory.objects.create(name="Test Me")
        Memory.objects.create(name="HR")
        agent = FakeAgent()

        run(agent, "--memory", str(a.id))

        assert list(agent.rags) == [str(a.id)]

    def test_an_unknown_memory_is_an_error(self):
        with pytest.raises(CommandError, match="not found"):
            run(FakeAgent(), "--memory", "00000000-0000-0000-0000-000000000000")
