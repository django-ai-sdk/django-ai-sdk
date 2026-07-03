"""
Unit tests for MemoryService permission enforcement.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================================================
# MemoryService — get_assistant_memories
# ============================================================================


@pytest.mark.django_db
@pytest.mark.asyncio
class TestMemoryServiceGetAssistantMemories:
    async def test_filters_by_slug(self):
        from django_ai_sdk.memories.models import Memory
        from django_ai_sdk.memories.services import MemoryService
        from tests.mocks.assistant import mock_assistant_memories

        mem1 = await Memory.objects.acreate(name="Legal Documents")
        await Memory.objects.acreate(name="General Docs")

        with mock_assistant_memories([mem1.slug]):
            result = await MemoryService.get_assistant_memories("test-asst")
            assert result == [str(mem1.id)]

    async def test_returns_empty_list_when_no_memories(self):
        from django_ai_sdk.memories.services import MemoryService
        from tests.mocks.assistant import mock_assistant_memories

        with mock_assistant_memories([]):
            result = await MemoryService.get_assistant_memories("test-asst")
            assert result == []

    async def test_skips_nonexistent_slugs(self):
        from django_ai_sdk.memories.services import MemoryService
        from tests.mocks.assistant import mock_assistant_memories

        with mock_assistant_memories(["nonexistent-slug"]):
            result = await MemoryService.get_assistant_memories("test-asst")
            assert result == []


# ============================================================================
# MemoryService — permission enforcement
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.django_db
class TestMemoryServicePermissions:
    """Integration tests: MemoryService methods enforce permissions."""

    async def _get_admin(self):
        from tests.factories.db import UserFactory

        return await UserFactory.acreate(is_superuser=True)

    async def _get_user(self):
        from tests.factories.db import UserFactory

        return await UserFactory.acreate()

    async def test_create_memory_requires_authenticated_user(self):
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.permissions import PermissionDenied

        with pytest.raises(PermissionDenied):
            await MemoryService.create_memory(
                name="test", user=None
            )

    async def test_owner_can_delete_memory(self):
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.memories.models import Memory, MemoryUser
        from tests.mocks.permissions import memory_permissions

        memory_user = await self._get_user()

        mem = await Memory.objects.acreate(
            name="to-delete", description="x", is_public=False
        )
        await MemoryUser.objects.acreate(memory=mem, user=memory_user, can_manage=True)

        with memory_permissions("django_ai_sdk.permissions.MemoryDefaultPermission"):
            await MemoryService.delete_memory(str(mem.id), user=memory_user)

        assert not await Memory.objects.filter(id=mem.id).aexists()

    async def test_stranger_cannot_delete_private_memory(self):
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.memories.models import Memory, MemoryUser
        from django_ai_sdk.permissions import PermissionDenied
        from tests.mocks.permissions import memory_permissions

        memory_user = await self._get_user()
        stranger = await self._get_user()

        mem = await Memory.objects.acreate(
            name="private", description="x", is_public=False
        )
        await MemoryUser.objects.acreate(memory=mem, user=memory_user, can_manage=True)

        with memory_permissions("django_ai_sdk.permissions.MemoryDefaultPermission"):
            with pytest.raises(PermissionDenied):
                await MemoryService.delete_memory(str(mem.id), user=stranger)

        await mem.adelete()

    async def test_stranger_can_read_public_memory(self):
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.memories.models import Memory, MemoryUser
        from tests.mocks.permissions import memory_permissions

        memory_user = await self._get_user()
        stranger = await self._get_user()

        mem = await Memory.objects.acreate(
            name="public", description="x", is_public=True
        )
        await MemoryUser.objects.acreate(memory=mem, user=memory_user, can_manage=True)

        with memory_permissions("django_ai_sdk.permissions.MemoryDefaultPermission"):
            result = await MemoryService.get_memory(str(mem.id), user=stranger)
            assert str(result.id) == str(mem.id)

        await mem.adelete()

    async def test_get_memory_denied_for_private_not_owner(self):
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.memories.models import Memory, MemoryUser
        from django_ai_sdk.permissions import PermissionDenied
        from tests.mocks.permissions import memory_permissions

        memory_user = await self._get_user()
        stranger = await self._get_user()

        mem = await Memory.objects.acreate(
            name="private-no-peek", description="x", is_public=False
        )
        await MemoryUser.objects.acreate(memory=mem, user=memory_user, can_manage=True)

        with memory_permissions("django_ai_sdk.permissions.MemoryDefaultPermission"):
            with pytest.raises(PermissionDenied):
                await MemoryService.get_memory(str(mem.id), user=stranger)

        await mem.adelete()

    async def test_link_memories_links_assistant_memories(self):
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.memories.models import Memory, MemoryUser, ThreadMemory
        from django_ai_sdk.conversation.models import Thread
        from tests.mocks.permissions import memory_permissions
        from tests.mocks.assistant import mock_assistant_memories

        memory_user = await self._get_user()
        thread = await Thread.objects.acreate()

        mem = await Memory.objects.acreate(
            name="link-test", is_public=False
        )
        await MemoryUser.objects.acreate(memory=mem, user=memory_user, can_manage=True)

        with (
            mock_assistant_memories([mem.slug]),
            memory_permissions("django_ai_sdk.permissions.MemoryDefaultPermission"),
        ):
            await MemoryService.link_memories(
                "test-asst", str(thread.id), user=memory_user
            )

        linked = await ThreadMemory.objects.filter(
            thread=thread, memory=mem
        ).aexists()
        assert linked

        await mem.adelete()

    async def test_link_memories_skips_when_user_cannot_read(self):
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.memories.models import Memory, MemoryUser, ThreadMemory
        from django_ai_sdk.conversation.models import Thread
        from tests.mocks.permissions import memory_permissions
        from tests.mocks.assistant import mock_assistant_memories

        memory_user = await self._get_user()
        thread = await Thread.objects.acreate()

        mem = await Memory.objects.acreate(
            name="req-link", is_public=False
        )
        await MemoryUser.objects.acreate(memory=mem, user=memory_user, can_manage=True)

        with (
            mock_assistant_memories([mem.slug]),
            memory_permissions("django_ai_sdk.permissions.MemoryDefaultPermission"),
        ):
            await MemoryService.link_memories(
                "test-asst", str(thread.id), user=None
            )

        linked = await ThreadMemory.objects.filter(
            thread=thread, memory=mem
        ).aexists()
        assert not linked

        await mem.adelete()

    async def test_unlink_memories_skips_when_user_cannot_read(self):
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.memories.models import Memory, MemoryUser, ThreadMemory
        from django_ai_sdk.conversation.models import Thread
        from tests.mocks.permissions import memory_permissions
        from tests.mocks.assistant import mock_assistant_memories

        memory_user = await self._get_user()
        thread = await Thread.objects.acreate()

        mem = await Memory.objects.acreate(
            name="req-unlink", is_public=False
        )
        await MemoryUser.objects.acreate(memory=mem, user=memory_user, can_manage=True)
        await ThreadMemory.objects.acreate(thread=thread, memory=mem, active=True)

        with (
            mock_assistant_memories([mem.slug]),
            memory_permissions("django_ai_sdk.permissions.MemoryDefaultPermission"),
        ):
            await MemoryService.unlink_memories(
                "test-asst", str(thread.id), user=None
            )

        linked = await ThreadMemory.objects.filter(
            thread=thread, memory=mem
        ).aexists()
        assert linked

        await mem.adelete()

    async def test_unlink_memories_unlinks_assistant_memories(self):
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.memories.models import Memory, MemoryUser, ThreadMemory
        from django_ai_sdk.conversation.models import Thread
        from tests.mocks.permissions import memory_permissions
        from tests.mocks.assistant import mock_assistant_memories

        memory_user = await self._get_user()
        thread = await Thread.objects.acreate()

        mem = await Memory.objects.acreate(
            name="unlink-test", is_public=False
        )
        await MemoryUser.objects.acreate(memory=mem, user=memory_user, can_manage=True)
        await ThreadMemory.objects.acreate(
            thread=thread, memory=mem, active=True
        )

        with (
            mock_assistant_memories([mem.slug]),
            memory_permissions("django_ai_sdk.permissions.MemoryDefaultPermission"),
        ):
            await MemoryService.unlink_memories(
                "test-asst", str(thread.id), user=memory_user
            )

        linked = await ThreadMemory.objects.filter(
            thread=thread, memory=mem
        ).aexists()
        assert not linked

        await mem.adelete()


    async def test_ownerless_memory_denied_all(self):
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.memories.models import Memory
        from django_ai_sdk.permissions import PermissionDenied
        from tests.mocks.permissions import memory_permissions

        user = await self._get_user()

        mem = await Memory.objects.acreate(
            name="ownerless", description="x", is_public=False
        )

        with memory_permissions("django_ai_sdk.permissions.MemoryDefaultPermission"):
            with pytest.raises(PermissionDenied):
                await MemoryService.get_memory(str(mem.id), user=user)

            with pytest.raises(PermissionDenied):
                await MemoryService.delete_memory(str(mem.id), user=user)

            with pytest.raises(PermissionDenied):
                await MemoryService.update_memory(str(mem.id), name="x", user=user)

        await mem.adelete()


# ============================================================================
# Open mode — AllowAll with user=None
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.django_db
class TestOpenMode:
    """Verify the SDK works without restriction under AllowAll + user=None."""

    async def test_thread_service_create_with_none_user(
        self, mock_assistants_registry, mock_storage_adapter_registry
    ):
        from django_ai_sdk.storage.services import ThreadService
        from tests.mocks.permissions import thread_permissions

        mock_storage_class = MagicMock()
        mock_storage_class.create_thread = AsyncMock(return_value="new-thread-id")
        mock_assistants_registry.get.return_value.storage_adapter = mock_storage_class

        with thread_permissions("django_ai_sdk.permissions.AllowAll"):
            result = await ThreadService.create_thread(
                "test-assistant", user=None
            )
        assert result == "new-thread-id"

    async def test_thread_service_get_with_none_user(
        self, mock_assistants_registry, mock_storage_adapter_registry
    ):
        from django_ai_sdk.storage.services import ThreadService
        from tests.mocks.permissions import thread_permissions
        from tests.mocks.storage import setup_thread_adapter

        thread_info, _ = setup_thread_adapter(
            mock_storage_adapter_registry, user_id=None
        )
        with thread_permissions("django_ai_sdk.permissions.AllowAll"):
            result = await ThreadService.get_thread("thread-1", user=None)
        assert result is not None
        assert result.assistant_id == "test-assistant"

    async def test_thread_service_storage_for_thread_with_none_user(
        self, mock_assistants_registry
    ):
        from django_ai_sdk.storage.services import ThreadService
        from tests.factories.schemas import ThreadInfoFactory
        from tests.mocks.permissions import thread_permissions

        thread_info = ThreadInfoFactory.build(assistant_id="test-assistant", user_id=None)

        with thread_permissions("django_ai_sdk.permissions.AllowAll"):
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
        from tests.mocks.permissions import thread_permissions
        from tests.mocks.storage import setup_thread_adapter, mock_get_storage

        setup_thread_adapter(mock_storage_adapter_registry, user_id=None)

        with thread_permissions("django_ai_sdk.permissions.AllowAll"):
            with mock_get_storage(method="rate_message", return_value=True):
                result = await ThreadService.rate_message(
                    "thread-1", "msg-1", 1, user=None
                )
        assert result is True

    async def test_memory_service_create_with_none_user(self):
        from django_ai_sdk.memories.services import MemoryService
        from tests.mocks.permissions import memory_permissions

        with memory_permissions("django_ai_sdk.permissions.AllowAll"):
            try:
                result = await MemoryService.create_memory(
                    name="open-test", user=None
                )
                assert result is not None
            finally:
                from django_ai_sdk.memories.models import Memory
                await Memory.objects.filter(name="open-test").adelete()

    async def test_memory_service_list_with_none_user(self):
        from django_ai_sdk.memories.services import MemoryService
        from tests.mocks.permissions import memory_permissions

        with memory_permissions("django_ai_sdk.permissions.AllowAll"):
            result = await MemoryService.list_memories(user=None)
            assert isinstance(result, list)


# ============================================================================
# MemoryUser Management — RBAC enforcement
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.django_db
class TestMemoryUserManagement:
    async def _get_user(self):
        from tests.factories.db import UserFactory

        return await UserFactory.acreate()

    async def test_owner_can_add_user(self):
        """Memory owner can add other users as viewers."""
        from django_ai_sdk.memories.models import Memory, MemoryUser
        from django_ai_sdk.memories.services import MemoryService
        from tests.mocks.permissions import memory_permissions

        owner = await self._get_user()
        viewer = await self._get_user()

        mem = await Memory.objects.acreate(name="shared", description="x")
        await MemoryUser.objects.acreate(memory=mem, user=owner, can_manage=True)

        with memory_permissions("django_ai_sdk.permissions.MemoryDefaultPermission"):
            result = await MemoryService.add_memory_user(
                str(mem.id), str(viewer.id), can_manage=False, user=owner
            )

        assert result.can_manage is False
        assert await MemoryUser.objects.filter(memory=mem, user=viewer).aexists()

    async def test_owner_can_promote_user_to_manager(self):
        """Memory owner can promote viewer to manager."""
        from django_ai_sdk.memories.models import Memory, MemoryUser
        from django_ai_sdk.memories.services import MemoryService
        from tests.mocks.permissions import memory_permissions

        owner = await self._get_user()
        member = await self._get_user()

        mem = await Memory.objects.acreate(name="promote", description="x")
        await MemoryUser.objects.acreate(memory=mem, user=owner, can_manage=True)
        await MemoryUser.objects.acreate(memory=mem, user=member, can_manage=False)

        with memory_permissions("django_ai_sdk.permissions.MemoryDefaultPermission"):
            result = await MemoryService.update_memory_user(
                str(mem.id), str(member.id), can_manage=True, user=owner
            )

        assert result.can_manage is True

    async def test_non_owner_cannot_add_user(self):
        """Stranger cannot add users to private memory."""
        from django_ai_sdk.memories.models import Memory, MemoryUser
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.permissions import PermissionDenied
        from tests.mocks.permissions import memory_permissions

        owner = await self._get_user()
        stranger = await self._get_user()
        viewer = await self._get_user()

        mem = await Memory.objects.acreate(name="private", description="x")
        await MemoryUser.objects.acreate(memory=mem, user=owner, can_manage=True)

        with memory_permissions("django_ai_sdk.permissions.MemoryDefaultPermission"):
            with pytest.raises(PermissionDenied):
                await MemoryService.add_memory_user(
                    str(mem.id), str(viewer.id), can_manage=False, user=stranger
                )

    async def test_viewer_cannot_add_user(self):
        """Non-manager member cannot add other users."""
        from django_ai_sdk.memories.models import Memory, MemoryUser
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.permissions import PermissionDenied
        from tests.mocks.permissions import memory_permissions

        owner = await self._get_user()
        viewer = await self._get_user()
        new_user = await self._get_user()

        mem = await Memory.objects.acreate(name="viewer-only", description="x")
        await MemoryUser.objects.acreate(memory=mem, user=owner, can_manage=True)
        await MemoryUser.objects.acreate(memory=mem, user=viewer, can_manage=False)

        with memory_permissions("django_ai_sdk.permissions.MemoryDefaultPermission"):
            with pytest.raises(PermissionDenied):
                await MemoryService.add_memory_user(
                    str(mem.id), str(new_user.id), can_manage=False, user=viewer
                )


@pytest.mark.asyncio
@pytest.mark.django_db
class TestMemoryServiceGetChunkContent:
    """get_chunk_content checks permissions on the Entry's parent Memory.

    MemoryDefaultPermission.has_object_permission expects a Memory (it reads
    memory_users/memory_groups, relations Entry doesn't have), raising an
    AttributeError instead of enforcing the permission.
    """

    async def _get_user(self):
        from tests.factories.db import UserFactory

        return await UserFactory.acreate()

    async def test_owner_can_read_chunk_content(self):
        from django_ai_sdk.memories.models import Entry, Memory, MemoryUser
        from django_ai_sdk.memories.services import MemoryService
        from tests.mocks.permissions import memory_permissions

        owner = await self._get_user()
        mem = await Memory.objects.acreate(name="chunk-owner", is_public=False)
        await MemoryUser.objects.acreate(memory=mem, user=owner, can_manage=True)
        entry = await Entry.objects.acreate(memory=mem, content="full content")

        with memory_permissions("django_ai_sdk.permissions.MemoryDefaultPermission"):
            result = await MemoryService.get_chunk_content(
                str(entry.id), None, user=owner
            )

        assert result == "full content"

    async def test_stranger_denied_chunk_content(self):
        from django_ai_sdk.memories.models import Entry, Memory, MemoryUser
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.permissions import PermissionDenied
        from tests.mocks.permissions import memory_permissions

        owner = await self._get_user()
        stranger = await self._get_user()
        mem = await Memory.objects.acreate(name="chunk-private", is_public=False)
        await MemoryUser.objects.acreate(memory=mem, user=owner, can_manage=True)
        entry = await Entry.objects.acreate(memory=mem, content="secret content")

        with memory_permissions("django_ai_sdk.permissions.MemoryDefaultPermission"):
            with pytest.raises(PermissionDenied):
                await MemoryService.get_chunk_content(str(entry.id), None, user=stranger)
