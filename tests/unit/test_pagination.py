"""
Pagination tests for service-layer limit/offset support.

Covers four distinct pagination paths:
  - MemoryStore (in-memory dict)
  - ThreadService.threads (multi-adapter merge then slice)
  - MemoryService.list_memories (DB queryset slice)
  - AgentService.list_agents (list slice on registry + DB)
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================================================
# MemoryStore — in-memory pagination
# ============================================================================


class TestMemoryStorePagination:
    def setup_method(self):
        from django_ai_sdk.storage.memory import MemoryStore

        MemoryStore.clear()

    def teardown_method(self):
        from django_ai_sdk.storage.memory import MemoryStore

        MemoryStore.clear()

    def _make_threads(self, n: int):
        from django_ai_sdk.storage.memory import MemoryStore

        for i in range(n):
            MemoryStore.create_thread(thread_id=f"t{i}", title=f"Thread {i}")

    def test_no_pagination_returns_all(self):
        from django_ai_sdk.storage.memory import MemoryStore

        self._make_threads(5)
        assert len(MemoryStore.list_threads()) == 5

    def test_limit_caps_results(self):
        from django_ai_sdk.storage.memory import MemoryStore

        self._make_threads(10)
        result = MemoryStore.list_threads(limit=3)
        assert len(result) == 3

    def test_offset_skips_items(self):
        from django_ai_sdk.storage.memory import MemoryStore

        self._make_threads(5)
        all_threads = MemoryStore.list_threads()
        result = MemoryStore.list_threads(offset=2)
        assert result == all_threads[2:]

    def test_limit_and_offset_together(self):
        from django_ai_sdk.storage.memory import MemoryStore

        self._make_threads(10)
        all_threads = MemoryStore.list_threads()
        result = MemoryStore.list_threads(limit=3, offset=4)
        assert result == all_threads[4:7]

    def test_offset_beyond_length_returns_empty(self):
        from django_ai_sdk.storage.memory import MemoryStore

        self._make_threads(3)
        result = MemoryStore.list_threads(offset=10, limit=5)
        assert result == []

    def test_limit_zero_returns_empty(self):
        from django_ai_sdk.storage.memory import MemoryStore

        self._make_threads(5)
        result = MemoryStore.list_threads(limit=0)
        assert result == []


# ============================================================================
# ThreadService.threads — multi-adapter merge then slice
# ============================================================================


def _make_thread_info(thread_id: str, age_seconds: int = 0):
    from django_ai_sdk.storage.schemas import ThreadInfo

    ts = datetime(2024, 1, 1, tzinfo=UTC) - timedelta(seconds=age_seconds)
    return ThreadInfo(
        id=thread_id,
        title=f"Thread {thread_id}",
        agent_id="test-agent",
        model="gpt-4",
        user_id=None,
        created_at=ts,
        updated_at=ts,
        metadata={},
        message_count=0,
    )


@pytest.mark.asyncio
class TestThreadServicePagination:
    async def _threads_via_mock_adapter(self, threads, limit=100, offset=0):
        """Run ThreadService.threads with a single mocked adapter returning `threads`."""
        from django_ai_sdk.storage.services import ThreadService

        mock_adapter = MagicMock()
        mock_adapter.__name__ = "MockAdapter"  # logger calls adapter_class.__name__
        mock_adapter.list_threads = AsyncMock(return_value=threads)

        with (
            patch(
                "django_ai_sdk.storage.services.StorageAdapterRegistry.get_all_adapters",
                return_value=[mock_adapter],
            ),
            patch.object(
                ThreadService,
                "has_perms",
                new_callable=AsyncMock,
            ),
        ):
            return await ThreadService.threads(user=None, limit=limit, offset=offset)

    async def test_default_limit_100(self):
        threads = [_make_thread_info(f"t{i}", age_seconds=i) for i in range(150)]
        result = await self._threads_via_mock_adapter(threads)
        assert len(result) == 100

    async def test_limit_caps_result(self):
        threads = [_make_thread_info(f"t{i}", age_seconds=i) for i in range(20)]
        result = await self._threads_via_mock_adapter(threads, limit=5)
        assert len(result) == 5

    async def test_offset_skips_items(self):
        threads = [_make_thread_info(f"t{i}", age_seconds=i) for i in range(10)]
        # Sorted newest first: t0 (age=0) comes first
        result_page1 = await self._threads_via_mock_adapter(threads, limit=5, offset=0)
        result_page2 = await self._threads_via_mock_adapter(threads, limit=5, offset=5)
        ids_page1 = [t.id for t in result_page1]
        ids_page2 = [t.id for t in result_page2]
        assert set(ids_page1) & set(ids_page2) == set()
        assert len(ids_page1) + len(ids_page2) == 10

    async def test_pagination_applied_after_sort(self):
        # Two adapters: merge + sort must happen before slice
        threads_a = [_make_thread_info("newest", age_seconds=0)]
        threads_b = [_make_thread_info("oldest", age_seconds=100)]

        mock_a = MagicMock()
        mock_a.__name__ = "MockAdapterA"
        mock_a.list_threads = AsyncMock(return_value=threads_a)
        mock_b = MagicMock()
        mock_b.__name__ = "MockAdapterB"
        mock_b.list_threads = AsyncMock(return_value=threads_b)

        from django_ai_sdk.storage.services import ThreadService

        with (
            patch(
                "django_ai_sdk.storage.services.StorageAdapterRegistry.get_all_adapters",
                return_value=[mock_a, mock_b],
            ),
            patch.object(
                ThreadService,
                "has_perms",
                new_callable=AsyncMock,
            ),
        ):
            result = await ThreadService.threads(user=None, limit=1, offset=0)

        assert len(result) == 1
        assert result[0].id == "newest"

    async def test_offset_beyond_total_returns_empty(self):
        threads = [_make_thread_info(f"t{i}") for i in range(3)]
        result = await self._threads_via_mock_adapter(threads, limit=10, offset=100)
        assert result == []


# ============================================================================
# MemoryService.list_memories — DB queryset slice
# ============================================================================


@pytest.mark.django_db
@pytest.mark.asyncio
class TestMemoryServiceListMemoriesPagination:
    async def _create_memories(self, n: int):
        from django_ai_sdk.memories.models import Memory

        import uuid

        memories = []
        for i in range(n):
            # Unique name per test run to avoid slug collisions across leaked rows
            m = await Memory.objects.acreate(name=f"PagTest {uuid.uuid4().hex[:8]}", is_public=True)
            memories.append(m)
        return memories

    async def test_limit_caps_results(self):
        from django_ai_sdk.memories.services import MemoryService

        await self._create_memories(5)
        # Use limit=2 — should always return exactly 2 regardless of other DB rows
        result = await MemoryService.list_memories(user=None, limit=2)
        assert len(result) == 2

    async def test_offset_shifts_page(self):
        from django_ai_sdk.memories.services import MemoryService

        await self._create_memories(5)
        # Use a limit large enough to get all rows in the DB
        all_results = await MemoryService.list_memories(user=None, limit=10_000)
        total = len(all_results)

        page2 = await MemoryService.list_memories(user=None, limit=10_000, offset=2)
        assert len(page2) == total - 2
        # First item after skip must match position [2] in the full list
        assert page2[0].id == all_results[2].id

    async def test_offset_beyond_count_returns_empty(self):
        from django_ai_sdk.memories.services import MemoryService

        await self._create_memories(3)
        result = await MemoryService.list_memories(user=None, limit=10, offset=100_000)
        assert result == []


# ============================================================================
# AgentService.list_agents — list slice (registry + DB)
# ============================================================================


@pytest.mark.asyncio
class TestAgentServiceListAgentsPagination:
    def _make_registry_with(self, n: int):
        """Return a mock registry with n visible agents."""
        mock_registry = MagicMock()
        agents = {}
        for i in range(n):
            a = MagicMock()
            a.name = f"Agent {i}"
            a.model = "gpt-4"
            a.permissions = []
            agents[f"asst-{i}"] = a
        mock_registry.visible.return_value = agents
        return mock_registry

    async def _list(self, n_registry: int, limit: int = 100, offset: int = 0):
        from django_ai_sdk.agents.services import AgentService
        from django_ai_sdk.agents.models import AgentSettings

        registry = self._make_registry_with(n_registry)

        with (
            patch("django_ai_sdk.agents.services.registry", registry),
            patch.object(
                AgentService,
                "has_perms",
                new_callable=AsyncMock,
            ),
            patch.object(
                AgentSettings.objects,
                "filter",
                return_value=MagicMock(__aiter__=MagicMock(return_value=aiter([]))),
            ),
        ):
            return await AgentService.list_agents(
                user=None, limit=limit, offset=offset
            )

    async def test_limit_caps_registry_results(self):
        result = await self._list(n_registry=10, limit=3)
        assert len(result) == 3

    async def test_offset_skips_registry_entries(self):
        all_results = await self._list(n_registry=10, limit=10, offset=0)
        page2 = await self._list(n_registry=10, limit=10, offset=3)
        assert len(page2) == len(all_results) - 3

    async def test_offset_beyond_count_returns_empty(self):
        result = await self._list(n_registry=5, limit=10, offset=100)
        assert result == []


def aiter(iterable):
    """Tiny helper: sync iterable → async iterator for mocking `async for`."""

    class _AsyncIter:
        def __init__(self, it):
            self._it = iter(it)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._it)
            except StopIteration:
                raise StopAsyncIteration

    return _AsyncIter(iterable)
