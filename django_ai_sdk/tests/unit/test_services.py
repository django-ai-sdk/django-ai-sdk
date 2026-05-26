"""
Unit tests for ThreadService and thread history service.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from django_ai_sdk.storage.services import (
    ThreadService,
    aget_thread_file_meta,
    aget_thread_history,
)
from django_ai_sdk.views.schemas import Message, MessagePart


def make_message(role: str, text: str, message_id: str = None) -> Message:
    return Message(
        role=role,
        parts=[MessagePart(type="text", text=text)],
        id=message_id,
    )


def make_mock_assistant(assistant_id="test-assistant", name="Test Assistant", model="gpt-4"):
    from django_ai_sdk.permissions import AllowAll
    from django_ai_sdk.storage.memory import MemoryStorageAdapter

    assistant = MagicMock()
    assistant.id = assistant_id
    assistant.name = name
    assistant.model = model
    assistant.storage_adapter = MemoryStorageAdapter
    assistant.permissions = [AllowAll]
    assistant.history = AsyncMock(
        return_value=MagicMock(thread={"id": "thread-1", "title": "Test"}, messages=[])
    )
    return assistant


@pytest.fixture
def mock_user():
    user = MagicMock()
    user.pk = "user-1"
    user.is_authenticated = True
    return user


@pytest.fixture
def mock_assistants_registry():
    assistant = make_mock_assistant()
    with \
        patch("django_ai_sdk.assistants.registry.registry") as reg, \
        patch("django_ai_sdk.assistants.services.registry", reg):
        reg.get = MagicMock(return_value=assistant)
        reg.all = MagicMock(return_value={"test-assistant": assistant})
        yield reg


@pytest.fixture
def mock_storage_adapter_registry():
    with patch("django_ai_sdk.storage.services.StorageAdapterRegistry") as sr:
        sr.get_all_adapters = MagicMock(return_value=[])
        yield sr


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
    async def test_creates_thread_with_auto_metadata(
        self, mock_assistants_registry, mock_user
    ):
        from django_ai_sdk.storage.memory import MemoryStorageAdapter

        with patch.object(MemoryStorageAdapter, "create_thread", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = "gen-thread-id"

            result = await ThreadService.create_thread(
                assistant_id="test-assistant",
                messages=[make_message("user", "Hello world")],
                user=mock_user,
                user_id="user-1",
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
                messages=[make_message("user", "Hello")],
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


@pytest.mark.asyncio
class TestThreadServiceRateMessage:
    async def test_rates_message_success(
        self, mock_assistants_registry, mock_storage_adapter_registry, mock_user
    ):
        thread_info = MagicMock(
            spec=["assistant_id", "user_id"],
            assistant_id="test-assistant",
            user_id="user-1",
        )
        mock_adapter_cls = make_mock_adapter_class(get_thread=thread_info)
        mock_storage_adapter_registry.get_all_adapters.return_value = [mock_adapter_cls]

        mock_storage = MagicMock()
        mock_storage.rate_message = AsyncMock(return_value=True)

        with patch(
            "django_ai_sdk.storage.services._get_storage",
            new_callable=AsyncMock,
        ) as mock_storage_internal:
            mock_storage_internal.return_value = mock_storage

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
        thread_info = MagicMock(
            spec=["assistant_id", "user_id"],
            assistant_id="test-assistant",
            user_id="user-1",
        )
        mock_adapter_cls = make_mock_adapter_class(get_thread=thread_info)
        mock_storage_adapter_registry.get_all_adapters.return_value = [mock_adapter_cls]

        mock_storage = MagicMock()
        mock_storage.rate_message = AsyncMock(return_value=False)

        with patch(
            "django_ai_sdk.storage.services._get_storage",
            new_callable=AsyncMock,
        ) as mock_storage_internal:
            mock_storage_internal.return_value = mock_storage

            with pytest.raises(ValueError, match="Message not found"):
                await ThreadService.rate_message(
                    "thread-1", "msg-1", 1, user=mock_user
                )


@pytest.mark.asyncio
class TestThreadServiceDeleteMessage:
    async def test_deletes_message_success(
        self, mock_assistants_registry, mock_storage_adapter_registry, mock_user
    ):
        thread_info = MagicMock(
            spec=["assistant_id", "user_id"],
            assistant_id="test-assistant",
            user_id="user-1",
        )
        mock_adapter_cls = make_mock_adapter_class(get_thread=thread_info)
        mock_storage_adapter_registry.get_all_adapters.return_value = [mock_adapter_cls]

        mock_storage = MagicMock()
        mock_storage.delete_message = AsyncMock(return_value=True)

        with patch(
            "django_ai_sdk.storage.services._get_storage",
            new_callable=AsyncMock,
        ) as mock_storage_internal:
            mock_storage_internal.return_value = mock_storage

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


@pytest.mark.asyncio
class TestThreadServiceRestoreMessage:
    async def test_restores_message_success(
        self, mock_assistants_registry, mock_storage_adapter_registry, mock_user
    ):
        thread_info = MagicMock(
            spec=["assistant_id", "user_id"],
            assistant_id="test-assistant",
            user_id="user-1",
        )
        mock_adapter_cls = make_mock_adapter_class(get_thread=thread_info)
        mock_storage_adapter_registry.get_all_adapters.return_value = [mock_adapter_cls]

        mock_storage = MagicMock()
        mock_storage.restore_message = AsyncMock(return_value=True)

        with patch(
            "django_ai_sdk.storage.services._get_storage",
            new_callable=AsyncMock,
        ) as mock_storage_internal:
            mock_storage_internal.return_value = mock_storage

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
# MemoryService Tests
# ============================================================================


@pytest.mark.django_db
@pytest.mark.asyncio
class TestMemoryServiceGetAssistantMemories:
    async def test_filters_by_slug(self):
        from django_ai_sdk.memories.models import Memory
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.tests.factories.memory_factory import MemoryFactory

        mem1 = await Memory.objects.acreate(name="Legal Documents")
        mem2 = await Memory.objects.acreate(name="Product Specs")
        slug1 = mem1.slug
        slug2 = mem2.slug

        mock_assistant = MagicMock()
        mock_assistant.memories = [slug1, slug2]

        with patch("django_ai_sdk.assistants.services.registry") as mock_reg:
            mock_reg.get.return_value = mock_assistant

            result = await MemoryService.get_assistant_memories("test-asst")

        assert sorted(result) == sorted([str(mem1.id), str(mem2.id)])

    async def test_returns_empty_when_no_memories_configured(self):
        from django_ai_sdk.memories.services import MemoryService

        mock_assistant = MagicMock()
        mock_assistant.memories = []

        with patch("django_ai_sdk.assistants.services.registry") as mock_reg:
            mock_reg.get.return_value = mock_assistant

            result = await MemoryService.get_assistant_memories("test-asst")

        assert result == []

    async def test_filters_by_slug_only_not_name(self):
        from django_ai_sdk.memories.models import Memory
        from django_ai_sdk.memories.services import MemoryService

        mem1 = await Memory.objects.acreate(name="Legal Documents")
        mem2 = await Memory.objects.acreate(name="Product Specs")
        # Filter by the other's name — should NOT match
        mock_assistant = MagicMock()
        mock_assistant.memories = [mem2.name]

        with patch("django_ai_sdk.assistants.services.registry") as mock_reg:
            mock_reg.get.return_value = mock_assistant

            result = await MemoryService.get_assistant_memories("test-asst")

        assert result == []


# ============================================================================
# Thread History Tests
# ============================================================================


@pytest.mark.asyncio
class TestGetThreadHistory:
    async def test_returns_thread_and_messages(
        self, mock_assistants_registry, mock_storage_adapter_registry, mock_user
    ):
        thread_info = MagicMock(
            spec=["assistant_id", "user_id"],
            assistant_id="test-assistant",
            user_id="user-1",
        )
        mock_adapter_cls = make_mock_adapter_class(get_thread=thread_info)
        mock_storage_adapter_registry.get_all_adapters.return_value = [mock_adapter_cls]

        result = await aget_thread_history("thread-1", user=mock_user)

        assert "thread" in result
        assert "messages" in result
        assert "memories" not in result
        assert "file_count" not in result
        assert "file_memory_id" not in result

    async def test_raises_when_thread_not_found(
        self, mock_storage_adapter_registry, mock_user
    ):
        mock_storage_adapter_registry.get_all_adapters.return_value = []

        with pytest.raises(ValueError, match="Thread not found"):
            await aget_thread_history("nonexistent", user=mock_user)


# ============================================================================
# Thread File Meta Tests
# ============================================================================


@pytest.mark.asyncio
class TestGetThreadFileMeta:
    async def test_returns_file_meta(self, mock_storage_adapter_registry):
        thread_db = MagicMock()
        thread_db.file_memory_id = None

        with patch("django_ai_sdk.conversation.models.Thread") as mock_thread:
            mock_thread.objects.filter.return_value.aexists = AsyncMock(return_value=True)
            mock_thread.objects.select_related.return_value.aget = AsyncMock(return_value=thread_db)

            result = await aget_thread_file_meta("thread-1")

            assert result["file_count"] == 0
            assert result["file_memory_id"] is None

    async def test_raises_when_thread_not_found(self):
        with patch("django_ai_sdk.conversation.models.Thread") as mock_thread:
            mock_thread.objects.filter.return_value.aexists = AsyncMock(return_value=False)

            with pytest.raises(ValueError, match="Thread not found"):
                await aget_thread_file_meta("nonexistent")

    async def test_counts_files_when_memory_exists(self, mock_storage_adapter_registry):
        thread_db = MagicMock()
        thread_db.file_memory_id = "mem-uuid-123"

        with patch("django_ai_sdk.conversation.models.Thread") as mock_thread, \
             patch("django_ai_sdk.memories.models.Entry") as mock_entry:
            mock_thread.objects.filter.return_value.aexists = AsyncMock(return_value=True)
            mock_thread.objects.select_related.return_value.aget = AsyncMock(return_value=thread_db)
            mock_entry.objects.filter.return_value.acount = AsyncMock(return_value=3)

            result = await aget_thread_file_meta("thread-1")

            assert result["file_count"] == 3
            assert result["file_memory_id"] == "mem-uuid-123"


# ============================================================================
# Permission Tests
# ============================================================================


@pytest.mark.asyncio
class TestPermissions:
    """Direct unit tests for the permission system classes."""

    async def test_allow_all_grants_permission(self):
        from django_ai_sdk.permissions import AllowAll, Operation, PermissionDenied, check_permissions

        await check_permissions(None, Operation.CHAT, [AllowAll])

    async def test_deny_all_denies_permission(self):
        from django_ai_sdk.permissions import DenyAll, Operation, PermissionDenied, check_permissions

        with pytest.raises(PermissionDenied):
            await check_permissions(None, Operation.CHAT, [DenyAll])

    async def test_is_authenticated_allows_authenticated(self):
        from django_ai_sdk.permissions import IsAuthenticated, Operation, PermissionDenied, check_permissions

        user = MagicMock(is_authenticated=True)
        await check_permissions(user, Operation.CHAT, [IsAuthenticated])

    async def test_is_authenticated_denies_anonymous(self):
        from django_ai_sdk.permissions import IsAuthenticated, Operation, PermissionDenied, check_permissions

        user = MagicMock(is_authenticated=False)
        with pytest.raises(PermissionDenied):
            await check_permissions(user, Operation.CHAT, [IsAuthenticated])

    async def test_is_authenticated_denies_none_user(self):
        from django_ai_sdk.permissions import IsAuthenticated, Operation, PermissionDenied, check_permissions

        with pytest.raises(PermissionDenied):
            await check_permissions(None, Operation.CHAT, [IsAuthenticated])

    async def test_is_admin_allows_staff(self):
        from django_ai_sdk.permissions import IsAdminUser, Operation, PermissionDenied, check_permissions

        user = MagicMock(is_staff=True)
        await check_permissions(user, Operation.CHAT, [IsAdminUser])

    async def test_is_admin_allows_superuser(self):
        from django_ai_sdk.permissions import IsAdminUser, Operation, PermissionDenied, check_permissions

        user = MagicMock(is_superuser=True)
        await check_permissions(user, Operation.CHAT, [IsAdminUser])

    async def test_is_admin_denies_regular_user(self):
        from django_ai_sdk.permissions import IsAdminUser, Operation, PermissionDenied, check_permissions

        user = MagicMock(is_staff=False, is_superuser=False)
        with pytest.raises(PermissionDenied):
            await check_permissions(user, Operation.CHAT, [IsAdminUser])

    async def test_is_owner_grants_when_user_id_matches(self):
        from django_ai_sdk.permissions import IsOwner, Operation, PermissionDenied, check_object_permissions

        user = MagicMock(pk="user-1")
        obj = MagicMock(user_id="user-1")
        await check_object_permissions(user, Operation.VIEW_THREAD, obj, [IsOwner])

    async def test_is_owner_denies_when_user_id_mismatch(self):
        from django_ai_sdk.permissions import IsOwner, Operation, PermissionDenied, check_object_permissions

        user = MagicMock(pk="user-1")
        obj = MagicMock(user_id="user-2")
        with pytest.raises(PermissionDenied):
            await check_object_permissions(user, Operation.VIEW_THREAD, obj, [IsOwner])

    async def test_is_owner_denies_none_user(self):
        from django_ai_sdk.permissions import IsOwner, Operation, PermissionDenied, check_object_permissions

        obj = MagicMock(user_id="user-1")
        with pytest.raises(PermissionDenied):
            await check_object_permissions(None, Operation.VIEW_THREAD, obj, [IsOwner])

    async def test_is_owner_allows_when_no_obj_user_id(self):
        """IsOwner allows if the object has no user_id (e.g. assistant-level check)."""
        from django_ai_sdk.permissions import IsOwner, Operation, PermissionDenied, check_permissions

        user = MagicMock(pk="user-1")
        await check_permissions(user, Operation.CREATE_THREAD, [IsOwner])

    async def test_is_owner_grants_all_for_memory_like_object(self):
        """IsOwner checks user_id, not owner_id — Memory objects only have owner_id,
        so IsOwner treats them as ownerless and grants all."""
        from django_ai_sdk.permissions import IsOwner, Operation, check_object_permissions

        user = MagicMock(pk="user-1")
        memory = MagicMock(spec=["owner_id"], owner_id="user-1")
        await check_object_permissions(user, Operation.VIEW_MEMORY, memory, [IsOwner])

    async def test_check_object_permissions_raises_on_first_failure(self):
        from django_ai_sdk.permissions import (
            AllowAll,
            DenyAll,
            Operation,
            PermissionDenied,
            check_object_permissions,
        )

        obj = MagicMock(user_id="user-1")
        user = MagicMock(id="user-2")
        with pytest.raises(PermissionDenied):
            await check_object_permissions(user, Operation.DELETE_THREAD, obj, [DenyAll, AllowAll])


@pytest.mark.asyncio
class TestThreadServiceCreateThreadPermissions:
    """Thread creation permission-denied scenarios."""

    async def test_denies_create_when_deny_all(self, mock_assistants_registry, mock_user):
        from django_ai_sdk.permissions import DenyAll, PermissionDenied

        reg = mock_assistants_registry
        reg.get.return_value.permissions = [DenyAll]

        with pytest.raises(PermissionDenied):
            await ThreadService.create_thread(
                assistant_id="test-assistant",
                messages=[],
                user=mock_user,
            )

    async def test_denies_create_when_not_authenticated(self, mock_assistants_registry):
        from django_ai_sdk.permissions import IsAuthenticated, PermissionDenied

        reg = mock_assistants_registry
        reg.get.return_value.permissions = [IsAuthenticated]
        anon_user = MagicMock(is_authenticated=False)

        with pytest.raises(PermissionDenied):
            await ThreadService.create_thread(
                assistant_id="test-assistant",
                messages=[],
                user=anon_user,
            )

    async def test_allows_create_when_authenticated(self, mock_assistants_registry, mock_user):
        from django_ai_sdk.permissions import IsAuthenticated

        reg = mock_assistants_registry
        reg.get.return_value.permissions = [IsAuthenticated]

        result = await ThreadService.create_thread(
            assistant_id="test-assistant",
            messages=[],
            user=mock_user,
        )
        assert result is not None


@pytest.mark.asyncio
class TestThreadServiceObjectPermissions:
    """Object-level permission scenarios (thread ownership, etc.)."""

    async def test_denies_rate_when_is_owner_and_mismatch(
        self, mock_assistants_registry, mock_storage_adapter_registry
    ):
        from django_ai_sdk.permissions import IsOwner, PermissionDenied

        reg = mock_assistants_registry
        reg.get.return_value.permissions = [IsOwner]

        thread_info = MagicMock(
            spec=["assistant_id", "user_id"],
            assistant_id="test-assistant",
            user_id="other-user",
        )
        mock_adapter_cls = make_mock_adapter_class(get_thread=thread_info)
        mock_storage_adapter_registry.get_all_adapters.return_value = [mock_adapter_cls]

        mock_storage = MagicMock()
        mock_storage.rate_message = AsyncMock(return_value=True)

        owner_user = MagicMock(pk="owner-user")

        with patch(
            "django_ai_sdk.storage.services._get_storage",
            new_callable=AsyncMock,
        ) as mock_storage_internal:
            mock_storage_internal.return_value = mock_storage

            with pytest.raises(PermissionDenied):
                await ThreadService.rate_message(
                    "thread-1", "msg-1", 1, user=owner_user
                )

    async def test_allows_rate_when_is_owner_and_matches(
        self, mock_assistants_registry, mock_storage_adapter_registry, mock_user
    ):
        from django_ai_sdk.permissions import IsOwner

        reg = mock_assistants_registry
        reg.get.return_value.permissions = [IsOwner]

        thread_info = MagicMock(
            spec=["assistant_id", "user_id"],
            assistant_id="test-assistant",
            user_id="user-1",
        )
        mock_adapter_cls = make_mock_adapter_class(get_thread=thread_info)
        mock_storage_adapter_registry.get_all_adapters.return_value = [mock_adapter_cls]

        mock_storage = MagicMock()
        mock_storage.rate_message = AsyncMock(return_value=True)

        with patch(
            "django_ai_sdk.storage.services._get_storage",
            new_callable=AsyncMock,
        ) as mock_storage_internal:
            mock_storage_internal.return_value = mock_storage

            result = await ThreadService.rate_message(
                "thread-1", "msg-1", 1, user=mock_user
            )
            assert result is True


# ============================================================================
# Memory Permission Tests
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.django_db
class TestMemoryDefaultPermission:
    """Test MemoryDefaultPermission class logic directly."""

    async def test_has_permission_allows_authenticated_for_create(self):
        from django_ai_sdk.permissions import MemoryDefaultPermission, Operation

        perm = MemoryDefaultPermission()
        user = MagicMock(is_authenticated=True)
        result = await perm.has_permission(user, Operation.CREATE_MEMORY)
        assert result is True

    async def test_has_permission_denies_anonymous_for_create(self):
        from django_ai_sdk.permissions import MemoryDefaultPermission, Operation

        perm = MemoryDefaultPermission()
        user = MagicMock(is_authenticated=False)
        result = await perm.has_permission(user, Operation.CREATE_MEMORY)
        assert result is False

    async def test_has_permission_denies_none_for_create(self):
        from django_ai_sdk.permissions import MemoryDefaultPermission, Operation

        perm = MemoryDefaultPermission()
        result = await perm.has_permission(None, Operation.CREATE_MEMORY)
        assert result is False

    async def test_has_permission_requires_auth_for_all_ops(self):
        from django_ai_sdk.permissions import MemoryDefaultPermission, Operation

        perm = MemoryDefaultPermission()
        user = MagicMock(is_authenticated=True)
        anon = MagicMock(is_authenticated=False)

        for op in Operation:
            result = await perm.has_permission(user, op)
            assert result is True, f"Auth user denied for {op}"
            result = await perm.has_permission(anon, op)
            assert result is False, f"Anon allowed for {op}"
            result = await perm.has_permission(None, op)
            assert result is False, f"None allowed for {op}"

    async def test_has_object_permission_grants_owner_all(self):
        from django_ai_sdk.permissions import MemoryDefaultPermission, Operation

        perm = MemoryDefaultPermission()
        user = MagicMock(pk="owner-1", is_authenticated=True)
        memory = MagicMock(
            owner_id="owner-1",
            is_public=False,
        )

        for op in Operation:
            result = await perm.has_object_permission(user, op, memory)
            assert result is True, f"Owner denied for {op}"

    async def test_has_object_permission_denies_non_contributor_private(self):
        from django_ai_sdk.permissions import MemoryDefaultPermission, Operation

        perm = MemoryDefaultPermission()
        user = MagicMock(pk="stranger", is_authenticated=True)
        mock_contrib = MagicMock()
        mock_contrib.filter.return_value.aexists = AsyncMock(return_value=False)
        memory = MagicMock(
            owner_id="owner-1",
            is_public=False,
            contributors=mock_contrib,
        )

        result = await perm.has_object_permission(user, Operation.VIEW_MEMORY, memory)
        assert result is False

    async def test_has_object_permission_allows_public_read_for_non_contributor(self):
        from django_ai_sdk.permissions import MemoryDefaultPermission, Operation

        perm = MemoryDefaultPermission()
        user = MagicMock(pk="stranger", is_authenticated=True)
        mock_contrib = MagicMock()
        mock_contrib.filter.return_value.aexists = AsyncMock(return_value=False)
        memory = MagicMock(
            owner_id="owner-1",
            is_public=True,
            contributors=mock_contrib,
        )

        result = await perm.has_object_permission(user, Operation.VIEW_MEMORY, memory)
        assert result is True

    async def test_has_object_permission_blocks_public_write_for_non_contributor(self):
        from django_ai_sdk.permissions import MemoryDefaultPermission, Operation

        perm = MemoryDefaultPermission()
        user = MagicMock(pk="stranger", is_authenticated=True)
        mock_contrib = MagicMock()
        mock_contrib.filter.return_value.aexists = AsyncMock(return_value=False)
        memory = MagicMock(
            owner_id="owner-1",
            is_public=True,
            contributors=mock_contrib,
        )

        result = await perm.has_object_permission(user, Operation.UPLOAD_DOCUMENT, memory)
        assert result is False

    async def test_has_object_permission_denies_anonymous_even_public(self):
        from django_ai_sdk.permissions import MemoryDefaultPermission, Operation

        perm = MemoryDefaultPermission()
        memory = MagicMock(
            owner_id="owner-1",
            is_public=True,
        )

        result = await perm.has_object_permission(None, Operation.VIEW_MEMORY, memory)
        assert result is False

    async def test_has_object_permission_denies_contributor_owner_ops(self):
        from django_ai_sdk.permissions import MemoryDefaultPermission, Operation

        perm = MemoryDefaultPermission()
        user = MagicMock(pk="contributor-1", is_authenticated=True)
        mock_contrib = MagicMock()
        mock_contrib.filter.return_value.aexists = AsyncMock(return_value=True)
        memory = MagicMock(
            owner_id="owner-1",
            is_public=True,
            contributors=mock_contrib,
        )

        result = await perm.has_object_permission(user, Operation.DELETE_MEMORY, memory)
        assert result is False

        result = await perm.has_object_permission(user, Operation.UPDATE_MEMORY, memory)
        assert result is False

    async def test_has_object_permission_grants_contributor_write_ops(self):
        from django_ai_sdk.permissions import MemoryDefaultPermission, Operation

        perm = MemoryDefaultPermission()
        user = MagicMock(pk="contributor-1", is_authenticated=True)
        mock_contrib = MagicMock()
        mock_contrib.filter.return_value.aexists = AsyncMock(return_value=True)
        memory = MagicMock(
            owner_id="owner-1",
            is_public=True,
            contributors=mock_contrib,
        )

        result = await perm.has_object_permission(user, Operation.UPLOAD_DOCUMENT, memory)
        assert result is True

        result = await perm.has_object_permission(user, Operation.DELETE_DOCUMENT, memory)
        assert result is True

    async def test_has_object_permission_grants_contributor_read_ops(self):
        from django_ai_sdk.permissions import MemoryDefaultPermission, Operation

        perm = MemoryDefaultPermission()
        user = MagicMock(pk="contributor-1", is_authenticated=True)
        mock_contrib = MagicMock()
        mock_contrib.filter.return_value.aexists = AsyncMock(return_value=True)
        memory = MagicMock(
            owner_id="owner-1",
            is_public=False,
            contributors=mock_contrib,
        )

        result = await perm.has_object_permission(user, Operation.VIEW_MEMORY, memory)
        assert result is True

    async def test_has_object_permission_denies_all_for_ownerless(self):
        from django_ai_sdk.permissions import MemoryDefaultPermission, Operation

        perm = MemoryDefaultPermission()
        user = MagicMock(pk="anyone", is_authenticated=True)
        memory = MagicMock(
            owner_id=None,
            is_public=True,
        )

        for op in Operation:
            result = await perm.has_object_permission(user, op, memory)
            assert result is False, f"Ownerless memory allowed for {op}"


@pytest.mark.asyncio
@pytest.mark.django_db
class TestMemoryServicePermissions:
    """Integration tests: MemoryService methods enforce permissions."""

    async def _get_admin(self):
        from django.contrib.auth import get_user_model

        User = get_user_model()
        user, _ = await User.objects.aupdate_or_create(
            id=1,
            defaults={"username": "admin", "is_superuser": True},
        )
        return user

    async def _get_user(self, pk, username):
        from django.contrib.auth import get_user_model

        User = get_user_model()
        user, _ = await User.objects.aupdate_or_create(
            id=pk,
            defaults={"username": username},
        )
        return user

    async def test_create_memory_requires_authenticated_user(self):
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.permissions import PermissionDenied

        with pytest.raises(PermissionDenied):
            await MemoryService.create_memory(
                name="test", user=None
            )

    async def test_owner_can_delete_memory(self):
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.memories.models import Memory
        from django.test.utils import override_settings

        owner = await self._get_user(20, "owner2")

        mem = await Memory.objects.acreate(
            name="to-delete", description="x", owner=owner, is_public=False
        )

        with override_settings(
            AI_SDK_MEMORY_PERMISSIONS=[
                "django_ai_sdk.permissions.MemoryDefaultPermission"
            ]
        ):
            from django_ai_sdk.memories.services import _get_memory_permissions
            _get_memory_permissions.cache_clear()

            await MemoryService.delete_memory(str(mem.id), user=owner)

            _get_memory_permissions.cache_clear()

        assert not await Memory.objects.filter(id=mem.id).aexists()

    async def test_stranger_cannot_delete_private_memory(self):
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.memories.models import Memory
        from django_ai_sdk.permissions import PermissionDenied
        from django.test.utils import override_settings

        owner = await self._get_user(30, "owner3")
        stranger = await self._get_user(31, "stranger3")

        mem = await Memory.objects.acreate(
            name="private", description="x", owner=owner, is_public=False
        )

        with override_settings(
            AI_SDK_MEMORY_PERMISSIONS=[
                "django_ai_sdk.permissions.MemoryDefaultPermission"
            ]
        ):
            from django_ai_sdk.memories.services import _get_memory_permissions
            _get_memory_permissions.cache_clear()

            with pytest.raises(PermissionDenied):
                await MemoryService.delete_memory(str(mem.id), user=stranger)

            _get_memory_permissions.cache_clear()

        await mem.adelete()

    async def test_stranger_can_read_public_memory(self):
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.memories.models import Memory
        from django.test.utils import override_settings

        owner = await self._get_user(40, "owner4")
        stranger = await self._get_user(41, "stranger4")

        mem = await Memory.objects.acreate(
            name="public", description="x", owner=owner, is_public=True
        )

        with override_settings(
            AI_SDK_MEMORY_PERMISSIONS=[
                "django_ai_sdk.permissions.MemoryDefaultPermission"
            ]
        ):
            from django_ai_sdk.memories.services import _get_memory_permissions
            _get_memory_permissions.cache_clear()

            result = await MemoryService.get_memory(str(mem.id), user=stranger)
            assert str(result.id) == str(mem.id)

            _get_memory_permissions.cache_clear()

        await mem.adelete()

    async def test_get_memory_denied_for_private_not_owner(self):
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.memories.models import Memory
        from django_ai_sdk.permissions import PermissionDenied
        from django.test.utils import override_settings

        owner = await self._get_user(50, "owner5")
        stranger = await self._get_user(51, "stranger5")

        mem = await Memory.objects.acreate(
            name="private-no-peek", description="x", owner=owner, is_public=False
        )

        with override_settings(
            AI_SDK_MEMORY_PERMISSIONS=[
                "django_ai_sdk.permissions.MemoryDefaultPermission"
            ]
        ):
            from django_ai_sdk.memories.services import _get_memory_permissions
            _get_memory_permissions.cache_clear()

            with pytest.raises(PermissionDenied):
                await MemoryService.get_memory(str(mem.id), user=stranger)

            _get_memory_permissions.cache_clear()

        await mem.adelete()

    async def test_link_memories_links_assistant_memories(self):
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.memories.models import Memory, ThreadMemory
        from django_ai_sdk.conversation.models import Thread
        from django.test.utils import override_settings
        from django_ai_sdk.assistants.services import registry

        owner = await self._get_user(60, "link-owner")
        thread = await Thread.objects.acreate()

        mem = await Memory.objects.acreate(
            name="link-test", owner=owner, is_public=False
        )

        mock_assistant = MagicMock()
        mock_assistant.memories = [mem.slug]

        with (
            patch.object(registry, "get", return_value=mock_assistant),
            override_settings(
                AI_SDK_MEMORY_PERMISSIONS=[
                    "django_ai_sdk.permissions.MemoryDefaultPermission"
                ]
            ),
        ):
            from django_ai_sdk.memories.services import _get_memory_permissions
            _get_memory_permissions.cache_clear()

            await MemoryService.link_memories(
                "test-asst", str(thread.id), user=owner
            )

            _get_memory_permissions.cache_clear()

        linked = await ThreadMemory.objects.filter(
            thread=thread, memory=mem
        ).aexists()
        assert linked

        await mem.adelete()

    async def test_unlink_memories_unlinks_assistant_memories(self):
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.memories.models import Memory, ThreadMemory
        from django_ai_sdk.conversation.models import Thread
        from django.test.utils import override_settings
        from django_ai_sdk.assistants.services import registry

        owner = await self._get_user(61, "unlink-owner")
        thread = await Thread.objects.acreate()

        mem = await Memory.objects.acreate(
            name="unlink-test", owner=owner, is_public=False
        )
        await ThreadMemory.objects.acreate(
            thread=thread, memory=mem, active=True
        )

        mock_assistant = MagicMock()
        mock_assistant.memories = [mem.slug]

        with (
            patch.object(registry, "get", return_value=mock_assistant),
            override_settings(
                AI_SDK_MEMORY_PERMISSIONS=[
                    "django_ai_sdk.permissions.MemoryDefaultPermission"
                ]
            ),
        ):
            from django_ai_sdk.memories.services import _get_memory_permissions
            _get_memory_permissions.cache_clear()

            await MemoryService.unlink_memories(
                "test-asst", str(thread.id), user=owner
            )

            _get_memory_permissions.cache_clear()

        linked = await ThreadMemory.objects.filter(
            thread=thread, memory=mem
        ).aexists()
        assert not linked

        await mem.adelete()

    async def test_link_memories_requires_user(self):
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.memories.models import Memory
        from django_ai_sdk.conversation.models import Thread
        from django_ai_sdk.permissions import PermissionDenied
        from django_ai_sdk.assistants.services import registry
        from django.test.utils import override_settings

        owner = await self._get_user(70, "link-user-req")
        thread = await Thread.objects.acreate()

        mem = await Memory.objects.acreate(
            name="req-link", owner=owner, is_public=False
        )

        mock_assistant = MagicMock()
        mock_assistant.memories = [mem.slug]

        with (
            patch.object(registry, "get", return_value=mock_assistant),
            override_settings(
                AI_SDK_MEMORY_PERMISSIONS=[
                    "django_ai_sdk.permissions.MemoryDefaultPermission"
                ]
            ),
        ):
            from django_ai_sdk.memories.services import _get_memory_permissions
            _get_memory_permissions.cache_clear()

            with pytest.raises(PermissionDenied):
                await MemoryService.link_memories(
                    "test-asst", str(thread.id), user=None
                )

            _get_memory_permissions.cache_clear()

        await mem.adelete()

    async def test_unlink_memories_requires_user(self):
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.memories.models import Memory
        from django_ai_sdk.conversation.models import Thread
        from django_ai_sdk.permissions import PermissionDenied
        from django_ai_sdk.assistants.services import registry
        from django.test.utils import override_settings

        owner = await self._get_user(71, "unlink-user-req")
        thread = await Thread.objects.acreate()

        mem = await Memory.objects.acreate(
            name="req-unlink", owner=owner, is_public=False
        )

        mock_assistant = MagicMock()
        mock_assistant.memories = [mem.slug]

        with (
            patch.object(registry, "get", return_value=mock_assistant),
            override_settings(
                AI_SDK_MEMORY_PERMISSIONS=[
                    "django_ai_sdk.permissions.MemoryDefaultPermission"
                ]
            ),
        ):
            from django_ai_sdk.memories.services import _get_memory_permissions
            _get_memory_permissions.cache_clear()

            with pytest.raises(PermissionDenied):
                await MemoryService.unlink_memories(
                    "test-asst", str(thread.id), user=None
                )

            _get_memory_permissions.cache_clear()

        await mem.adelete()

    async def test_ownerless_memory_denied_all(self):
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.memories.models import Memory
        from django_ai_sdk.permissions import PermissionDenied
        from django.test.utils import override_settings

        user = await self._get_user(80, "anyone")

        mem = await Memory.objects.acreate(
            name="ownerless", description="x", owner=None, is_public=True
        )

        with override_settings(
            AI_SDK_MEMORY_PERMISSIONS=[
                "django_ai_sdk.permissions.MemoryDefaultPermission"
            ]
        ):
            from django_ai_sdk.memories.services import _get_memory_permissions
            _get_memory_permissions.cache_clear()

            with pytest.raises(PermissionDenied):
                await MemoryService.get_memory(str(mem.id), user=user)

            with pytest.raises(PermissionDenied):
                await MemoryService.delete_memory(str(mem.id), user=user)

            with pytest.raises(PermissionDenied):
                await MemoryService.update_memory(str(mem.id), name="x", user=user)

            _get_memory_permissions.cache_clear()

        await mem.adelete()


@pytest.mark.asyncio
@pytest.mark.django_db
class TestOpenMode:
    """Verify the SDK works without restriction under AllowAll + user=None."""

    async def test_thread_service_create_with_none_user(
        self, mock_assistants_registry, mock_storage_adapter_registry
    ):
        from django_ai_sdk.storage.services import ThreadService

        mock_storage_class = MagicMock()
        mock_storage_class.create_thread = AsyncMock(return_value="new-thread-id")
        mock_assistants_registry.get.return_value.storage_adapter = mock_storage_class

        result = await ThreadService.create_thread(
            "test-assistant", messages=[], user=None
        )
        assert result == "new-thread-id"

    async def test_thread_service_get_with_none_user(
        self, mock_assistants_registry, mock_storage_adapter_registry
    ):
        from django_ai_sdk.storage.services import ThreadService

        thread_info = MagicMock(
            spec=["assistant_id", "user_id"],
            assistant_id="test-assistant",
            user_id=None,
        )
        mock_adapter_cls = make_mock_adapter_class(get_thread=thread_info)
        mock_storage_adapter_registry.get_all_adapters.return_value = [mock_adapter_cls]

        result = await ThreadService.get_thread("thread-1", user=None)
        assert result is not None
        assert result.assistant_id == "test-assistant"

    async def test_thread_service_storage_for_thread_with_none_user(
        self, mock_assistants_registry
    ):
        from django_ai_sdk.storage.services import ThreadService, _get_thread, _get_storage

        thread_info = MagicMock(
            spec=["assistant_id", "user_id"],
            assistant_id="test-assistant",
            user_id=None,
        )

        with (
            patch(
                "django_ai_sdk.storage.services._get_thread",
                return_value=thread_info,
            ),
            patch(
                "django_ai_sdk.storage.services._get_storage",
                return_value=MagicMock(),
            ),
        ):
            result = await ThreadService.storage_for_thread("thread-1", user=None)
            assert result is not None

    async def test_thread_service_rate_with_none_user(
        self, mock_assistants_registry, mock_storage_adapter_registry
    ):
        from django_ai_sdk.storage.services import ThreadService

        thread_info = MagicMock(
            spec=["assistant_id", "user_id"],
            assistant_id="test-assistant",
            user_id=None,
        )
        mock_adapter_cls = make_mock_adapter_class(get_thread=thread_info)
        mock_storage_adapter_registry.get_all_adapters.return_value = [mock_adapter_cls]

        mock_storage = MagicMock()
        mock_storage.rate_message = AsyncMock(return_value=True)

        with patch(
            "django_ai_sdk.storage.services._get_storage",
            new_callable=AsyncMock,
        ) as mock_storage_internal:
            mock_storage_internal.return_value = mock_storage

            result = await ThreadService.rate_message(
                "thread-1", "msg-1", 1, user=None
            )
            assert result is True

    async def test_memory_service_create_with_none_user(self):
        from django_ai_sdk.memories.services import MemoryService, _get_memory_permissions
        from django.test.utils import override_settings

        with override_settings(AI_SDK_MEMORY_PERMISSIONS=[]):
            _get_memory_permissions.cache_clear()

            try:
                result = await MemoryService.create_memory(
                    name="open-test", user=None
                )
                assert result is not None
            finally:
                from django_ai_sdk.memories.models import Memory
                await Memory.objects.filter(name="open-test").adelete()
                _get_memory_permissions.cache_clear()

    async def test_memory_service_list_with_none_user(self):
        from django_ai_sdk.memories.services import MemoryService, _get_memory_permissions
        from django.test.utils import override_settings

        with override_settings(AI_SDK_MEMORY_PERMISSIONS=[]):
            _get_memory_permissions.cache_clear()

            result = await MemoryService.list_memories(user=None)
            assert isinstance(result, list)

            _get_memory_permissions.cache_clear()
