"""
Unit tests for ThreadService and thread history service.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from django_ai_sdk.views.schemas import Message, MessagePart
from django_ai_sdk.storage.services import (
    ThreadService,
    aget_thread_history,
)


def make_message(role: str, text: str, message_id: str = None) -> Message:
    return Message(
        role=role,
        parts=[MessagePart(type="text", text=text)],
        id=message_id,
    )


def make_mock_assistant(assistant_id="test-assistant", name="Test Assistant", model="gpt-4"):
    from django_ai_sdk.storage.memory import MemoryStorageAdapter

    assistant = MagicMock()
    assistant.id = assistant_id
    assistant.name = name
    assistant.model = model
    assistant.storage_adapter = MemoryStorageAdapter
    assistant.history = AsyncMock(
        return_value=MagicMock(thread={"id": "thread-1", "title": "Test"}, messages=[])
    )
    return assistant


@pytest.fixture
def mock_assistants_registry():
    assistant = make_mock_assistant()
    with patch("django_ai_sdk.assistants.registry.registry") as reg:
        reg.get = MagicMock(return_value=assistant)
        reg.all = MagicMock(return_value={"test-assistant": assistant})
        yield reg


@pytest.fixture
def mock_storage_adapter_registry():
    with patch("django_ai_sdk.storage.services.StorageAdapterRegistry") as sr:
        sr.get_all_adapters = MagicMock(return_value=[])
        yield sr


class AsyncEmptyIterator:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


def make_empty_query():
    q = MagicMock()
    q.__aiter__ = lambda s: AsyncEmptyIterator()
    return q


def make_mock_adapter_class(get_thread=None):
    adapter_cls = MagicMock()
    adapter_cls.__name__ = "MockAdapter"
    if get_thread is not None:
        adapter_cls.get_thread = AsyncMock(return_value=get_thread)
    else:
        adapter_cls.get_thread = AsyncMock(return_value=None)
    return adapter_cls


# ============================================================================
# ThreadService Tests
# ============================================================================


@pytest.mark.asyncio
class TestThreadServiceCreateThread:
    async def test_creates_thread_with_auto_metadata(self, mock_assistants_registry):
        from django_ai_sdk.storage.memory import MemoryStorageAdapter

        with patch.object(MemoryStorageAdapter, "create_thread", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = "gen-thread-id"

            result = await ThreadService.create_thread(
                assistant_id="test-assistant",
                messages=[make_message("user", "Hello world")],
                user_id="user-1",
            )

            assert result == "gen-thread-id"
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["metadata"]["assistant_id"] == "test-assistant"
            assert call_kwargs["metadata"]["model"] == "gpt-4"
            assert call_kwargs["metadata"]["assistant_name"] == "Test Assistant"
            assert call_kwargs["metadata"]["created_via"] == "create_thread"

    async def test_auto_generates_title_from_messages(self, mock_assistants_registry):
        from django_ai_sdk.storage.memory import MemoryStorageAdapter

        with patch.object(MemoryStorageAdapter, "create_thread", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = "gen-thread-id"

            await ThreadService.create_thread(
                assistant_id="test-assistant",
                messages=[make_message("user", "A" * 60)],
            )

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["title"] == "A" * 50 + "..."

    async def test_raises_when_assistant_not_found(self, mock_assistants_registry):
        mock_assistants_registry.get.return_value = None

        with pytest.raises(ValueError, match="Assistant not found"):
            await ThreadService.create_thread(assistant_id="nonexistent")


@pytest.mark.asyncio
class TestThreadServiceRateMessage:
    async def test_rates_message_success(self, mock_assistants_registry, mock_storage_adapter_registry):
        thread_info = MagicMock(assistant_id="test-assistant")
        mock_adapter_cls = make_mock_adapter_class(get_thread=thread_info)
        mock_storage_adapter_registry.get_all_adapters.return_value = [mock_adapter_cls]

        mock_storage = MagicMock()
        mock_storage.rate_message = AsyncMock(return_value=True)

        with patch.object(ThreadService, "get_storage") as mock_get_storage:
            mock_storage_cls = MagicMock()
            mock_storage_cls.return_value = mock_storage
            mock_get_storage.return_value = mock_storage_cls

            result = await ThreadService.rate_message("thread-1", "msg-1", 1)
            assert result is True

    async def test_raises_when_thread_not_found(self, mock_storage_adapter_registry):
        mock_storage_adapter_registry.get_all_adapters.return_value = []

        with pytest.raises(ValueError, match="Thread not found"):
            await ThreadService.rate_message("nonexistent", "msg-1", 1)

    async def test_raises_when_message_not_found(self, mock_assistants_registry, mock_storage_adapter_registry):
        thread_info = MagicMock(assistant_id="test-assistant")
        mock_adapter_cls = make_mock_adapter_class(get_thread=thread_info)
        mock_storage_adapter_registry.get_all_adapters.return_value = [mock_adapter_cls]

        mock_storage = MagicMock()
        mock_storage.rate_message = AsyncMock(return_value=False)

        with patch.object(ThreadService, "get_storage") as mock_get_storage:
            mock_storage_cls = MagicMock()
            mock_storage_cls.return_value = mock_storage
            mock_get_storage.return_value = mock_storage_cls

            with pytest.raises(ValueError, match="Message not found"):
                await ThreadService.rate_message("thread-1", "msg-1", 1)


@pytest.mark.asyncio
class TestThreadServiceDeleteMessage:
    async def test_deletes_message_success(self, mock_assistants_registry, mock_storage_adapter_registry):
        thread_info = MagicMock(assistant_id="test-assistant")
        mock_adapter_cls = make_mock_adapter_class(get_thread=thread_info)
        mock_storage_adapter_registry.get_all_adapters.return_value = [mock_adapter_cls]

        mock_storage = MagicMock()
        mock_storage.delete_message = AsyncMock(return_value=True)

        with patch.object(ThreadService, "get_storage") as mock_get_storage:
            mock_storage_cls = MagicMock()
            mock_storage_cls.return_value = mock_storage
            mock_get_storage.return_value = mock_storage_cls

            result = await ThreadService.delete_message("thread-1", "msg-1")
            assert result is True

    async def test_raises_when_message_not_found(self, mock_storage_adapter_registry):
        mock_storage_adapter_registry.get_all_adapters.return_value = []

        with pytest.raises(ValueError, match="Thread not found"):
            await ThreadService.delete_message("nonexistent", "msg-1")


@pytest.mark.asyncio
class TestThreadServiceRestoreMessage:
    async def test_restores_message_success(self, mock_assistants_registry, mock_storage_adapter_registry):
        thread_info = MagicMock(assistant_id="test-assistant")
        mock_adapter_cls = make_mock_adapter_class(get_thread=thread_info)
        mock_storage_adapter_registry.get_all_adapters.return_value = [mock_adapter_cls]

        mock_storage = MagicMock()
        mock_storage.restore_message = AsyncMock(return_value=True)

        with patch.object(ThreadService, "get_storage") as mock_get_storage:
            mock_storage_cls = MagicMock()
            mock_storage_cls.return_value = mock_storage
            mock_get_storage.return_value = mock_storage_cls

            result = await ThreadService.restore_message("thread-1", "msg-1")
            assert result is True

    async def test_raises_when_message_not_found(self, mock_storage_adapter_registry):
        mock_storage_adapter_registry.get_all_adapters.return_value = []

        with pytest.raises(ValueError, match="Thread not found"):
            await ThreadService.restore_message("nonexistent", "msg-1")


# ============================================================================
# Thread History Tests
# ============================================================================


@pytest.mark.asyncio
class TestGetThreadHistory:
    async def test_returns_thread_data(self, mock_assistants_registry, mock_storage_adapter_registry):
        thread_db = MagicMock()
        thread_db.file_memory_id = None

        mock_adapter_cls = make_mock_adapter_class(
            get_thread=MagicMock(assistant_id="test-assistant")
        )
        mock_storage_adapter_registry.get_all_adapters.return_value = [mock_adapter_cls]

        with patch("django_ai_sdk.conversation.models.Thread") as mock_thread, \
             patch("django_ai_sdk.memories.models.ThreadMemory") as mock_thread_memory, \
             patch("django_ai_sdk.memories.models.Entry") as mock_entry:
            mock_thread.objects.select_related.return_value.aget = AsyncMock(return_value=thread_db)
            mock_thread_memory.objects.filter.return_value.select_related.return_value.annotate.return_value = make_empty_query()
            mock_entry.objects.filter.return_value.acount = AsyncMock(return_value=0)

            result = await aget_thread_history("thread-1")

            assert "thread" in result
            assert "memories" in result
            assert "messages" in result
            assert "file_count" in result
            assert "file_memory_id" in result

    async def test_raises_when_thread_not_found(self, mock_storage_adapter_registry):
        mock_storage_adapter_registry.get_all_adapters.return_value = []

        with pytest.raises(ValueError, match="Thread not found"):
            await aget_thread_history("nonexistent")

    async def test_raises_when_assistant_not_found(self, mock_storage_adapter_registry):
        mock_storage_adapter_registry.get_all_adapters.return_value = []

        with pytest.raises(ValueError, match="Thread not found"):
            await aget_thread_history("thread-1")
