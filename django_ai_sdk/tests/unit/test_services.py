"""
Unit tests for the view services layer.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from django_ai_sdk.views.schemas import ChatRequest, Message, MessagePart
from django_ai_sdk.views.services import (
    aadd_message_to_thread,
    acreate_thread,
    adelete_all_threads,
    adelete_message,
    adelete_thread,
    aget_assistant_info,
    aget_thread_history,
    alist_assistants,
    alist_threads,
    areindex_assistant,
    arestore_message,
    arate_message,
    create_thread,
    delete_thread,
    list_threads,
)


def make_message(role: str, text: str, message_id: str = None) -> Message:
    return Message(
        role=role,
        parts=[MessagePart(type="text", text=text)],
        id=message_id,
    )


def make_mock_assistant(assistant_id="test-assistant", name="Test Assistant", model="gpt-4"):
    assistant = MagicMock()
    assistant.id = assistant_id
    assistant.name = name
    assistant.model = model
    assistant.history = AsyncMock(
        return_value=MagicMock(thread={"id": "thread-1", "title": "Test"}, messages=[])
    )
    assistant.as_view = AsyncMock(return_value=MagicMock())
    storage = MagicMock()
    storage.rate_message = AsyncMock(return_value=True)
    storage.delete_message = AsyncMock(return_value=True)
    storage.restore_message = AsyncMock(return_value=True)
    assistant.get_storage_adapter = AsyncMock(return_value=storage)
    assistant.info = MagicMock(return_value={"id": assistant_id, "name": name, "model": model})
    return assistant


@pytest.fixture
def mock_assistant():
    return make_mock_assistant()


@pytest.fixture
def mock_registry(mock_assistant):
    with patch("django_ai_sdk.views.services.registry") as reg:
        reg.get = MagicMock(return_value=mock_assistant)
        reg.all = MagicMock(return_value={"test-assistant": mock_assistant})
        yield reg


@pytest.fixture
def mock_thread_service():
    with patch("django_ai_sdk.views.services.ThreadService") as ts:
        ts.get_assistant = AsyncMock()
        ts.threads = AsyncMock(return_value=[])
        ts.create_thread = AsyncMock(return_value="thread-new-123")
        ts.delete_thread = AsyncMock(return_value=True)
        ts.delete_all_threads = AsyncMock(return_value=5)
        yield ts


@pytest.fixture
def mock_thread_model():
    with patch("django_ai_sdk.views.services.Thread") as mt:
        mt.objects = MagicMock()
        mt.objects.select_related.return_value.aget = AsyncMock()
        yield mt


class AsyncEmptyIterator:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


def make_empty_query():
    q = MagicMock()
    q.__aiter__ = lambda s: AsyncEmptyIterator()
    return q


@pytest.fixture
def mock_thread_memory_model():
    with patch("django_ai_sdk.views.services.ThreadMemory") as mtm:
        mtm.objects = MagicMock()
        mtm.objects.filter.return_value.select_related.return_value.annotate.return_value = make_empty_query()
        yield mtm


@pytest.fixture
def mock_entry_model():
    with patch("django_ai_sdk.views.services.Entry") as me:
        me.objects = MagicMock()
        me.objects.filter.return_value.acount = AsyncMock(return_value=0)
        yield me


@pytest.mark.asyncio
class TestAListThreads:
    async def test_returns_list_of_dicts(self, mock_thread_service):
        thread = MagicMock()
        thread.id = "t1"
        thread.title = "Test"
        thread.assistant_id = "a1"
        thread.created_at.isoformat.return_value = "2026-01-01T00:00:00"
        thread.updated_at.isoformat.return_value = "2026-01-02T00:00:00"
        thread.message_count = 3
        mock_thread_service.threads = AsyncMock(return_value=[thread])

        result = await alist_threads("user-1")

        assert len(result) == 1
        assert result[0]["id"] == "t1"
        assert result[0]["title"] == "Test"
        assert result[0]["message_count"] == 3

    async def test_empty_list(self, mock_thread_service):
        mock_thread_service.threads = AsyncMock(return_value=[])

        result = await alist_threads("user-1")
        assert result == []


class TestListThreadsSync:
    def test_sync_wrapper_calls_async(self, mock_thread_service):
        mock_thread_service.threads = AsyncMock(return_value=[])

        result = list_threads("user-1")
        assert result == []


@pytest.mark.asyncio
class TestACreateThread:
    async def test_creates_thread_with_default_title(self, mock_registry, mock_thread_service):
        payload = ChatRequest(messages=[], assistant_id="test-assistant")
        result = await acreate_thread(payload, "user-1")

        assert result == "thread-new-123"
        mock_thread_service.create_thread.assert_called_once()
        call_kwargs = mock_thread_service.create_thread.call_args.kwargs
        assert call_kwargs["title"] == "New Conversation"

    async def test_creates_thread_with_generated_title(self, mock_registry, mock_thread_service):
        long_text = "A" * 60
        payload = ChatRequest(
            messages=[make_message("user", long_text)],
            assistant_id="test-assistant",
        )
        result = await acreate_thread(payload, "user-1")

        assert result == "thread-new-123"
        call_kwargs = mock_thread_service.create_thread.call_args.kwargs
        assert call_kwargs["title"] == "A" * 50 + "..."

    async def test_raises_when_no_assistant_id(self, mock_registry):
        payload = ChatRequest(messages=[], assistant_id=None)

        with pytest.raises(ValueError, match="assistant_id is required"):
            await acreate_thread(payload, "user-1")

    async def test_raises_when_assistant_not_found(self, mock_registry):
        mock_registry.get = MagicMock(return_value=None)

        payload = ChatRequest(messages=[], assistant_id="nonexistent")

        with pytest.raises(ValueError, match="Assistant 'nonexistent' not found"):
            await acreate_thread(payload, "user-1")


@pytest.mark.asyncio
class TestAGetThreadHistory:
    async def test_returns_thread_data(self, mock_registry, mock_thread_service, mock_thread_model, mock_thread_memory_model, mock_entry_model):
        thread_obj = MagicMock(assistant_id="test-assistant")
        mock_thread_service.get_assistant = AsyncMock(return_value=thread_obj)

        thread_db = MagicMock()
        thread_db.file_memory_id = None
        mock_thread_model.objects.select_related.return_value.aget = AsyncMock(return_value=thread_db)

        result = await aget_thread_history("thread-1")

        assert "thread" in result
        assert "memories" in result
        assert "messages" in result
        assert "file_count" in result
        assert "file_memory_id" in result

    async def test_raises_when_thread_not_found(self, mock_thread_service):
        mock_thread_service.get_assistant = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="Thread not found"):
            await aget_thread_history("nonexistent")

    async def test_raises_when_assistant_not_found(self, mock_registry, mock_thread_service):
        mock_registry.get = MagicMock(return_value=None)
        thread_obj = MagicMock(assistant_id="gone-assistant")
        mock_thread_service.get_assistant = AsyncMock(return_value=thread_obj)

        with pytest.raises(ValueError, match="Assistant 'gone-assistant' not found"):
            await aget_thread_history("thread-1")


@pytest.mark.asyncio
class TestAAddMessageToThread:
    async def test_calls_assistant_view(self, mock_registry, mock_thread_service):
        thread_obj = MagicMock(assistant_id="test-assistant")
        mock_thread_service.get_assistant = AsyncMock(return_value=thread_obj)

        await aadd_message_to_thread("thread-1", [], MagicMock())

        mock_registry.get.return_value.as_view.assert_called_once()


@pytest.mark.asyncio
class TestADeleteThread:
    async def test_returns_true(self, mock_thread_service):
        mock_thread_service.delete_thread = AsyncMock(return_value=True)

        result = await adelete_thread("thread-1")
        assert result is True

    async def test_returns_false(self, mock_thread_service):
        mock_thread_service.delete_thread = AsyncMock(return_value=False)

        result = await adelete_thread("nonexistent")
        assert result is False


class TestDeleteThreadSync:
    def test_sync_wrapper(self, mock_thread_service):
        mock_thread_service.delete_thread = AsyncMock(return_value=True)

        assert delete_thread("thread-1") is True


@pytest.mark.asyncio
class TestADeleteAllThreads:
    async def test_returns_count(self, mock_thread_service):
        mock_thread_service.delete_all_threads = AsyncMock(return_value=10)

        result = await adelete_all_threads()
        assert result == 10


@pytest.mark.asyncio
class TestARateMessage:
    async def test_rates_message(self, mock_registry, mock_thread_service):
        thread_obj = MagicMock(assistant_id="test-assistant")
        mock_thread_service.get_assistant = AsyncMock(return_value=thread_obj)

        result = await arate_message("thread-1", "msg-1", 1)

        assert result == {"id": "msg-1", "rating": 1, "is_deleted": False}

    async def test_raises_when_message_not_found(self, mock_registry, mock_thread_service):
        thread_obj = MagicMock(assistant_id="test-assistant")
        mock_thread_service.get_assistant = AsyncMock(return_value=thread_obj)
        mock_registry.get.return_value.get_storage_adapter.return_value.rate_message = AsyncMock(return_value=False)

        with pytest.raises(ValueError, match="Message not found"):
            await arate_message("thread-1", "msg-1", 1)


@pytest.mark.asyncio
class TestADeleteMessage:
    async def test_deletes_message(self, mock_registry, mock_thread_service):
        thread_obj = MagicMock(assistant_id="test-assistant")
        mock_thread_service.get_assistant = AsyncMock(return_value=thread_obj)

        result = await adelete_message("thread-1", "msg-1")

        assert result == {"id": "msg-1", "is_deleted": True}

    async def test_raises_when_message_not_found(self, mock_registry, mock_thread_service):
        thread_obj = MagicMock(assistant_id="test-assistant")
        mock_thread_service.get_assistant = AsyncMock(return_value=thread_obj)
        mock_registry.get.return_value.get_storage_adapter.return_value.delete_message = AsyncMock(return_value=False)

        with pytest.raises(ValueError, match="Message not found"):
            await adelete_message("thread-1", "msg-1")


@pytest.mark.asyncio
class TestARestoreMessage:
    async def test_restores_message(self, mock_registry, mock_thread_service):
        thread_obj = MagicMock(assistant_id="test-assistant")
        mock_thread_service.get_assistant = AsyncMock(return_value=thread_obj)

        result = await arestore_message("thread-1", "msg-1")

        assert result == {"id": "msg-1", "is_deleted": False}

    async def test_raises_when_message_not_found(self, mock_registry, mock_thread_service):
        thread_obj = MagicMock(assistant_id="test-assistant")
        mock_thread_service.get_assistant = AsyncMock(return_value=thread_obj)
        mock_registry.get.return_value.get_storage_adapter.return_value.restore_message = AsyncMock(return_value=False)

        with pytest.raises(ValueError, match="Message not found"):
            await arestore_message("thread-1", "msg-1")


@pytest.mark.asyncio
class TestAListAssistants:
    async def test_returns_list(self, mock_registry):
        result = await alist_assistants()
        assert len(result) == 1
        assert result[0]["id"] == "test-assistant"
        assert result[0]["name"] == "Test Assistant"
        assert result[0]["model"] == "gpt-4"


@pytest.mark.asyncio
class TestAGetAssistantInfo:
    async def test_returns_info(self, mock_registry):
        result = await aget_assistant_info("test-assistant")
        assert result["id"] == "test-assistant"
        assert result["name"] == "Test Assistant"

    async def test_raises_when_not_found(self, mock_registry):
        mock_registry.get = MagicMock(return_value=None)

        with pytest.raises(ValueError, match="Assistant 'missing' not found"):
            await aget_assistant_info("missing")


@pytest.mark.asyncio
class TestAReindexAssistant:
    async def test_reindex_success(self, mock_registry):
        with patch("django_ai_sdk.views.services.Assistant") as mock_assistant_cls:
            mock_assistant_cls.reindex = AsyncMock(return_value=True)

            result = await areindex_assistant("test-assistant")
            assert result["success"] is True
            assert "reindexed successfully" in result["message"]

    async def test_reindex_no_rag_provider(self, mock_registry):
        with patch("django_ai_sdk.views.services.Assistant") as mock_assistant_cls:
            mock_assistant_cls.reindex = AsyncMock(return_value=None)

            result = await areindex_assistant("test-assistant")
            assert result["success"] is False
            assert "No RAG provider" in result["message"]

    async def test_raises_when_assistant_not_found(self, mock_registry):
        mock_registry.get = MagicMock(return_value=None)

        with pytest.raises(ValueError, match="Assistant 'missing' not found"):
            await areindex_assistant("missing")

    async def test_force_rebuild_message(self, mock_registry):
        with patch("django_ai_sdk.views.services.Assistant") as mock_assistant_cls:
            mock_assistant_cls.reindex = AsyncMock(return_value=True)

            result = await areindex_assistant("test-assistant", force_rebuild=True)
            assert "force rebuild" in result["message"]

    async def test_memory_id_in_message(self, mock_registry):
        with patch("django_ai_sdk.views.services.Assistant") as mock_assistant_cls:
            mock_assistant_cls.reindex = AsyncMock(return_value=True)

            result = await areindex_assistant("test-assistant", memory_id="mem-1")
            assert "mem-1" in result["message"]
