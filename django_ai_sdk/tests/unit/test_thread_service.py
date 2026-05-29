"""
Unit tests for ThreadService operations.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from django_ai_sdk.storage.services import (
    ThreadService,
    aget_thread_file_meta,
    aget_thread_history,
)
from django_ai_sdk.tests.factories.schemas import chat_message


# ============================================================================
# ThreadService — create
# ============================================================================


@pytest.mark.asyncio
class TestThreadServiceCreateThread:
    async def test_creates_thread_with_auto_metadata(
        self, mock_assistants_registry, mock_user
    ):
        from django_ai_sdk.storage.memory import MemoryStorageAdapter

        with patch.object(MemoryStorageAdapter, "create_thread", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = "gen-thread-id"

            result = await ThreadService.create_thread(
                assistant_id="test-assistant",
                messages=[chat_message("user", "Hello world")],
                user=mock_user,
            )

            assert result == "gen-thread-id"
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["metadata"]["assistant_id"] == "test-assistant"
            assert call_kwargs["metadata"]["model"] == "gpt-4"
            assert call_kwargs["metadata"]["assistant_name"] == "Test Assistant"
            assert call_kwargs["metadata"]["created_via"] == "create_thread"

    async def test_create_thread_passes_empty_title(
        self, mock_assistants_registry, mock_user
    ):
        from django_ai_sdk.storage.memory import MemoryStorageAdapter

        with patch.object(MemoryStorageAdapter, "create_thread", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = "gen-thread-id"

            await ThreadService.create_thread(
                assistant_id="test-assistant",
                messages=[chat_message("user", "Hello")],
                user=mock_user,
            )

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["title"] == ""

    async def test_raises_when_assistant_not_found(
        self, mock_assistants_registry, mock_user
    ):
        mock_assistants_registry.get.return_value = None

        with pytest.raises(ValueError, match="Assistant not found"):
            await ThreadService.create_thread(
                assistant_id="nonexistent", user=mock_user
            )


# ============================================================================
# ThreadService — rate_message
# ============================================================================


@pytest.mark.asyncio
class TestThreadServiceRateMessage:
    async def test_rates_message_success(
        self, mock_assistants_registry, mock_storage_adapter_registry, mock_user
    ):
        from django_ai_sdk.tests.mocks.storage import setup_thread_adapter, mock_get_storage

        setup_thread_adapter(mock_storage_adapter_registry)

        with mock_get_storage(method="rate_message", return_value=True):
            result = await ThreadService.rate_message(
                "thread-1", "msg-1", 1, user=mock_user
            )
            assert result is True

    async def test_raises_when_thread_not_found(
        self, mock_storage_adapter_registry, mock_user
    ):
        mock_storage_adapter_registry.get_all_adapters.return_value = []

        with pytest.raises(ValueError, match="Thread not found"):
            await ThreadService.rate_message(
                "nonexistent", "msg-1", 1, user=mock_user
            )

    async def test_raises_when_message_not_found(
        self, mock_assistants_registry, mock_storage_adapter_registry, mock_user
    ):
        from django_ai_sdk.tests.mocks.storage import setup_thread_adapter, mock_get_storage

        setup_thread_adapter(mock_storage_adapter_registry)

        with mock_get_storage(method="rate_message", return_value=False):
            with pytest.raises(ValueError, match="Message not found"):
                await ThreadService.rate_message(
                    "thread-1", "msg-1", 1, user=mock_user
                )


# ============================================================================
# ThreadService — delete_message
# ============================================================================


@pytest.mark.asyncio
class TestThreadServiceDeleteMessage:
    async def test_deletes_message_success(
        self, mock_assistants_registry, mock_storage_adapter_registry, mock_user
    ):
        from django_ai_sdk.tests.mocks.storage import setup_thread_adapter, mock_get_storage

        setup_thread_adapter(mock_storage_adapter_registry)

        with mock_get_storage(method="delete_message", return_value=True):
            result = await ThreadService.delete_message(
                "thread-1", "msg-1", user=mock_user
            )
            assert result is True

    async def test_raises_when_message_not_found(
        self, mock_storage_adapter_registry, mock_user
    ):
        mock_storage_adapter_registry.get_all_adapters.return_value = []

        with pytest.raises(ValueError, match="Thread not found"):
            await ThreadService.delete_message(
                "nonexistent", "msg-1", user=mock_user
            )


# ============================================================================
# ThreadService — restore_message
# ============================================================================


@pytest.mark.asyncio
class TestThreadServiceRestoreMessage:
    async def test_restores_message_success(
        self, mock_assistants_registry, mock_storage_adapter_registry, mock_user
    ):
        from django_ai_sdk.tests.mocks.storage import setup_thread_adapter, mock_get_storage

        setup_thread_adapter(mock_storage_adapter_registry)

        with mock_get_storage(method="restore_message", return_value=True):
            result = await ThreadService.restore_message(
                "thread-1", "msg-1", user=mock_user
            )
            assert result is True

    async def test_raises_when_message_not_found(
        self, mock_storage_adapter_registry, mock_user
    ):
        mock_storage_adapter_registry.get_all_adapters.return_value = []

        with pytest.raises(ValueError, match="Thread not found"):
            await ThreadService.restore_message(
                "nonexistent", "msg-1", user=mock_user
            )


# ============================================================================
# ThreadService — create_thread permission checks
# ============================================================================


@pytest.mark.asyncio
class TestThreadServiceCreateThreadPermissions:
    """Thread creation permission-denied scenarios."""

    async def test_denies_create_when_deny_all(
        self, assistant_permissions, mock_user
    ):
        from django_ai_sdk.permissions import DenyAll, PermissionDenied

        assistant_permissions(DenyAll)

        with pytest.raises(PermissionDenied):
            await ThreadService.create_thread(
                assistant_id="test-assistant", user=mock_user
            )

    async def test_denies_create_when_not_authenticated(
        self, assistant_permissions
    ):
        from django_ai_sdk.permissions import IsAuthenticated, PermissionDenied

        assistant_permissions(IsAuthenticated)

        with pytest.raises(PermissionDenied):
            await ThreadService.create_thread(
                assistant_id="test-assistant", user=None
            )

    async def test_allows_create_when_authenticated(
        self, assistant_permissions, mock_user
    ):
        from django_ai_sdk.permissions import IsAuthenticated

        assistant_permissions(IsAuthenticated)

        thread_id = await ThreadService.create_thread(
            assistant_id="test-assistant", user=mock_user
        )
        assert thread_id is not None


# ============================================================================
# ThreadService — object-level permissions
# ============================================================================


@pytest.mark.asyncio
class TestThreadServiceObjectPermissions:
    """Thread service methods that delegate to object-level permissions."""

    async def test_denies_rate_when_is_owner_and_mismatch(
        self, assistant_permissions, mock_storage_adapter_registry, mock_user
    ):
        from django_ai_sdk.permissions import IsOwner, PermissionDenied
        from django_ai_sdk.tests.mocks.storage import setup_thread_adapter

        assistant_permissions(IsOwner)
        setup_thread_adapter(mock_storage_adapter_registry, user_id="other-user")

        with pytest.raises(PermissionDenied):
            await ThreadService.rate_message(
                "thread-1", "msg-1", 1, user=mock_user
            )

    async def test_allows_rate_when_is_owner_and_matches(
        self, assistant_permissions, mock_storage_adapter_registry, mock_user
    ):
        from django_ai_sdk.permissions import IsOwner
        from django_ai_sdk.tests.mocks.storage import setup_thread_adapter, mock_get_storage

        assistant_permissions(IsOwner)
        setup_thread_adapter(mock_storage_adapter_registry)

        with mock_get_storage(method="rate_message", return_value=True):
            result = await ThreadService.rate_message(
                "thread-1", "msg-1", 1, user=mock_user
            )
            assert result is True


# ============================================================================
# Thread history & file meta
# ============================================================================


@pytest.mark.django_db
@pytest.mark.asyncio
class TestGetThreadHistory:
    async def test_returns_thread_and_messages(
        self, mock_assistants_registry, mock_storage_adapter_registry
    ):
        from django_ai_sdk.tests.mocks.storage import setup_thread_adapter

        setup_thread_adapter(mock_storage_adapter_registry)
        result = await aget_thread_history("thread-1")
        assert result["thread"] is not None

    async def test_raises_when_thread_not_found(
        self, mock_storage_adapter_registry
    ):
        from django_ai_sdk.tests.mocks.assistant import create_mock_adapter_class

        mock_adapter_cls = create_mock_adapter_class(get_thread=None)
        mock_storage_adapter_registry.get_all_adapters.return_value = [mock_adapter_cls]

        with pytest.raises(ValueError, match="not found"):
            await aget_thread_history("nonexistent")


@pytest.mark.django_db
@pytest.mark.asyncio
class TestGetThreadFileMeta:
    async def test_returns_file_meta(self):
        from django_ai_sdk.tests.mocks.storage import mock_thread_model

        file_memory_id = str(uuid4())
        thread_db = MagicMock()
        thread_db.file_memory_id = file_memory_id

        with mock_thread_model(aexists=True, thread_db=thread_db):
            result = await aget_thread_file_meta("thread-1")
            assert result["file_memory_id"] == file_memory_id
            assert "file_count" in result

    async def test_raises_when_thread_not_found(self):
        from django_ai_sdk.tests.mocks.storage import mock_thread_model

        with mock_thread_model(aexists=False):
            with pytest.raises(ValueError, match="not found"):
                await aget_thread_file_meta("nonexistent")

    async def test_counts_files_when_memory_exists(self):
        from unittest.mock import patch
        from django_ai_sdk.tests.mocks.storage import mock_thread_model

        file_memory_id = str(uuid4())
        thread_db = MagicMock()
        thread_db.file_memory_id = file_memory_id

        mock_entry_qs = MagicMock()
        mock_entry_qs.acount = AsyncMock(return_value=5)

        with (
            mock_thread_model(aexists=True, thread_db=thread_db),
            patch("django_ai_sdk.memories.models.Entry") as mock_entry,
        ):
            mock_entry.objects.filter.return_value = mock_entry_qs

            result = await aget_thread_file_meta("thread-1")
            assert result["file_count"] == 5
