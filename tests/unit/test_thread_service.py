"""
Unit tests for ThreadService operations.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from django_ai_sdk.common import THREAD_TITLE_MAX_LENGTH
from django_ai_sdk.conversation.models import Thread
from django_ai_sdk.storage.db import DbStorageAdapter
from django_ai_sdk.storage.services import (
    ThreadService,
    aget_thread_file_meta,
    aget_thread_history,
)
from tests.factories.schemas import chat_message


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
                user=mock_user,
            )

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["title"] == ""

    async def test_raises_when_assistant_not_found(
        self, mock_assistants_registry, mock_user
    ):
        mock_assistants_registry.get.return_value = None

        with pytest.raises(ValueError, match="not found"):
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
        from tests.mocks.storage import setup_thread_adapter, mock_get_storage

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
        from tests.mocks.storage import setup_thread_adapter, mock_get_storage

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
        from tests.mocks.storage import setup_thread_adapter, mock_get_storage

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
        from tests.mocks.storage import setup_thread_adapter, mock_get_storage

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
        from tests.mocks.storage import setup_thread_adapter

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
        from tests.mocks.storage import setup_thread_adapter, mock_get_storage

        assistant_permissions(IsOwner)
        setup_thread_adapter(mock_storage_adapter_registry)

        with mock_get_storage(method="rate_message", return_value=True):
            result = await ThreadService.rate_message(
                "thread-1", "msg-1", 1, user=mock_user
            )
            assert result is True


# ============================================================================
# DbStorageAdapter — update_thread title truncation
# ============================================================================


@pytest.mark.django_db
class TestThreadTitleMaxLengthConstant:
    def test_constant_matches_db_column(self):
        """Guards against the constant and the model field drifting apart."""
        assert THREAD_TITLE_MAX_LENGTH == Thread._meta.get_field("title").max_length


@pytest.mark.django_db
@pytest.mark.asyncio
class TestDbStorageAdapterUpdateThreadTitle:
    """`title` must never exceed the DB column's max_length.

    `update_thread` saves via `thread.asave()`, which bypasses Django's
    `full_clean()` validation - an over-length title would otherwise hit the
    raw DB constraint and raise instead of being handled gracefully.
    """

    async def test_truncates_title_exceeding_max_length(self):
        thread = await Thread.objects.acreate()
        overlong_title = "x" * (THREAD_TITLE_MAX_LENGTH + 50)

        result = await DbStorageAdapter.update_thread(str(thread.id), title=overlong_title)

        assert result is True
        await thread.arefresh_from_db()
        assert thread.title == overlong_title[:THREAD_TITLE_MAX_LENGTH]
        assert len(thread.title) == THREAD_TITLE_MAX_LENGTH

    async def test_leaves_title_within_max_length_untouched(self):
        thread = await Thread.objects.acreate()

        result = await DbStorageAdapter.update_thread(str(thread.id), title="Short title")

        assert result is True
        await thread.arefresh_from_db()
        assert thread.title == "Short title"


# ============================================================================
# Thread history & file meta
# ============================================================================


@pytest.mark.django_db
@pytest.mark.asyncio
class TestGetThreadHistory:
    async def test_returns_thread_and_messages(
        self, mock_assistants_registry, mock_storage_adapter_registry, mock_user
    ):
        from tests.mocks.storage import setup_thread_adapter

        setup_thread_adapter(mock_storage_adapter_registry)
        result = await aget_thread_history("thread-1", user=mock_user)
        assert result["thread"] is not None

    async def test_raises_when_thread_not_found(
        self, mock_storage_adapter_registry, mock_user
    ):
        from tests.mocks.assistant import create_mock_adapter_class

        mock_adapter_cls = create_mock_adapter_class(get_thread=None)
        mock_storage_adapter_registry.get_all_adapters.return_value = [mock_adapter_cls]

        with pytest.raises(ValueError, match="not found"):
            await aget_thread_history("nonexistent", user=mock_user)


@pytest.mark.django_db
@pytest.mark.asyncio
class TestGetThreadFileMeta:
    async def test_returns_file_meta(self):
        file_memory_id = str(uuid4())

        with (
            patch("django_ai_sdk.storage.services._get_thread", return_value=MagicMock(assistant_id="test")),
            patch("django_ai_sdk.conversation.models.Thread") as mock_thread,
        ):
            mock_thread.objects.filter.return_value.values_list.return_value.afirst = AsyncMock(return_value=file_memory_id)

            result = await aget_thread_file_meta("thread-1", user=None)
            assert result["file_memory_id"] == file_memory_id
            assert "file_count" in result

    async def test_raises_when_thread_not_found(self):
        with patch("django_ai_sdk.storage.services._get_thread", return_value=None):
            with pytest.raises(ValueError, match="not found"):
                await aget_thread_file_meta("nonexistent", user=None)

    async def test_counts_files_when_memory_exists(self):
        file_memory_id = str(uuid4())
        mock_entry_qs = MagicMock()
        mock_entry_qs.acount = AsyncMock(return_value=5)

        with (
            patch("django_ai_sdk.storage.services._get_thread", return_value=MagicMock(assistant_id="test")),
            patch("django_ai_sdk.conversation.models.Thread") as mock_thread,
            patch("django_ai_sdk.memories.models.Entry") as mock_entry,
        ):
            mock_thread.objects.filter.return_value.values_list.return_value.afirst = AsyncMock(return_value=file_memory_id)
            mock_entry.objects.filter.return_value = mock_entry_qs

            result = await aget_thread_file_meta("thread-1", user=None)
            assert result["file_count"] == 5
