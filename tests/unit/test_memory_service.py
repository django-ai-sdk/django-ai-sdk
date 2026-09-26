"""
Unit tests for MemoryService permission enforcement.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from django.core.files.base import ContentFile

import pytest


# ============================================================================
# MemoryService — get_agent_memories
# ============================================================================


@pytest.mark.django_db
@pytest.mark.asyncio
class TestMemoryServiceGetAgentMemories:
    async def test_filters_by_slug(self):
        from django_ai_sdk.memories.models import Memory
        from django_ai_sdk.memories.services import MemoryService
        from tests.mocks.agent import mock_agent_memories

        mem1 = await Memory.objects.acreate(name="Legal Documents")
        await Memory.objects.acreate(name="General Docs")

        with mock_agent_memories([mem1.slug]):
            result = await MemoryService.get_agent_memories("test-asst")
            assert result == [str(mem1.id)]

    async def test_returns_empty_list_when_no_memories(self):
        from django_ai_sdk.memories.services import MemoryService
        from tests.mocks.agent import mock_agent_memories

        with mock_agent_memories([]):
            result = await MemoryService.get_agent_memories("test-asst")
            assert result == []

    async def test_skips_nonexistent_slugs(self):
        from django_ai_sdk.memories.services import MemoryService
        from tests.mocks.agent import mock_agent_memories

        with mock_agent_memories(["nonexistent-slug"]):
            result = await MemoryService.get_agent_memories("test-asst")
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

    async def test_link_memories_links_agent_memories(self):
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.memories.models import Memory, MemoryUser, ThreadMemory
        from django_ai_sdk.conversation.models import Thread
        from tests.mocks.permissions import memory_permissions
        from tests.mocks.agent import mock_agent_memories

        memory_user = await self._get_user()
        thread = await Thread.objects.acreate()

        mem = await Memory.objects.acreate(
            name="link-test", is_public=False
        )
        await MemoryUser.objects.acreate(memory=mem, user=memory_user, can_manage=True)

        with (
            mock_agent_memories([mem.slug]),
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
        from tests.mocks.agent import mock_agent_memories

        memory_user = await self._get_user()
        thread = await Thread.objects.acreate()

        mem = await Memory.objects.acreate(
            name="req-link", is_public=False
        )
        await MemoryUser.objects.acreate(memory=mem, user=memory_user, can_manage=True)

        with (
            mock_agent_memories([mem.slug]),
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
        from tests.mocks.agent import mock_agent_memories

        memory_user = await self._get_user()
        thread = await Thread.objects.acreate()

        mem = await Memory.objects.acreate(
            name="req-unlink", is_public=False
        )
        await MemoryUser.objects.acreate(memory=mem, user=memory_user, can_manage=True)
        await ThreadMemory.objects.acreate(thread=thread, memory=mem, active=True)

        with (
            mock_agent_memories([mem.slug]),
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

    async def test_unlink_memories_unlinks_agent_memories(self):
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.memories.models import Memory, MemoryUser, ThreadMemory
        from django_ai_sdk.conversation.models import Thread
        from tests.mocks.permissions import memory_permissions
        from tests.mocks.agent import mock_agent_memories

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
            mock_agent_memories([mem.slug]),
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
        self, mock_agents_registry, mock_storage_adapter_registry
    ):
        from django_ai_sdk.storage.services import ThreadService
        from tests.mocks.permissions import thread_permissions

        mock_storage_class = MagicMock()
        mock_storage_class.create_thread = AsyncMock(return_value="new-thread-id")
        mock_agents_registry.get.return_value.storage_adapter = mock_storage_class

        with thread_permissions("django_ai_sdk.permissions.AllowAll"):
            result = await ThreadService.create_thread(
                "test-agent", user=None
            )
        assert result == "new-thread-id"

    async def test_thread_service_get_with_none_user(
        self, mock_agents_registry, mock_storage_adapter_registry
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
        assert result.agent_id == "test-agent"

    async def test_thread_service_storage_for_thread_with_none_user(
        self, mock_agents_registry
    ):
        from django_ai_sdk.storage.services import ThreadService
        from tests.factories.schemas import ThreadInfoFactory
        from tests.mocks.permissions import thread_permissions

        thread_info = ThreadInfoFactory.build(agent_id="test-agent", user_id=None)

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
        self, mock_agents_registry, mock_storage_adapter_registry
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


# ============================================================================
# MemoryService — list_thread_memories
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.django_db
class TestMemoryServiceListThreadMemories:
    """list_thread_memories enforces LIST_THREAD_MEMORIES permission."""

    async def _get_user(self):
        from tests.factories.db import UserFactory

        return await UserFactory.acreate()

    async def test_owner_can_list_thread_memories(self):
        from django_ai_sdk.memories.models import Memory, MemoryUser, ThreadMemory
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.conversation.models import Thread
        from tests.mocks.permissions import memory_permissions

        owner = await self._get_user()
        thread = await Thread.objects.acreate(user=owner)
        mem = await Memory.objects.acreate(name="owned", is_public=False)
        await MemoryUser.objects.acreate(memory=mem, user=owner, can_manage=True)
        await ThreadMemory.objects.acreate(thread=thread, memory=mem, active=True)

        with memory_permissions("django_ai_sdk.permissions.MemoryDefaultPermission"):
            result = await MemoryService.list_thread_memories(str(thread.id), user=owner)

        assert len(result) == 1
        assert str(result[0].id) == str(mem.id)

    async def test_stranger_cannot_list_private_thread_memory(self):
        from django_ai_sdk.memories.models import Memory, MemoryUser, ThreadMemory
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.conversation.models import Thread
        from tests.mocks.permissions import memory_permissions

        owner = await self._get_user()
        stranger = await self._get_user()
        thread = await Thread.objects.acreate(user=stranger)
        mem = await Memory.objects.acreate(name="private", is_public=False)
        await MemoryUser.objects.acreate(memory=mem, user=owner, can_manage=True)
        await ThreadMemory.objects.acreate(thread=thread, memory=mem, active=True)

        with memory_permissions("django_ai_sdk.permissions.MemoryDefaultPermission"):
            result = await MemoryService.list_thread_memories(str(thread.id), user=stranger)

        assert len(result) == 0

    async def test_stranger_can_list_public_thread_memory(self):
        from django_ai_sdk.memories.models import Memory, ThreadMemory
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.conversation.models import Thread
        from tests.mocks.permissions import memory_permissions

        stranger = await self._get_user()
        thread = await Thread.objects.acreate(user=stranger)
        mem = await Memory.objects.acreate(name="public", is_public=True)
        await ThreadMemory.objects.acreate(thread=thread, memory=mem, active=True)

        with memory_permissions("django_ai_sdk.permissions.MemoryDefaultPermission"):
            result = await MemoryService.list_thread_memories(str(thread.id), user=stranger)

        assert len(result) == 1
        assert str(result[0].id) == str(mem.id)

    async def test_anonymous_is_refused(self):
        from django_ai_sdk.memories.models import Memory, ThreadMemory
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.conversation.models import Thread
        from django_ai_sdk.permissions import PermissionDenied
        from tests.mocks.permissions import memory_permissions

        thread = await Thread.objects.acreate()
        mem = await Memory.objects.acreate(name="public", is_public=True)
        await ThreadMemory.objects.acreate(thread=thread, memory=mem, active=True)

        with (
            memory_permissions("django_ai_sdk.permissions.MemoryDefaultPermission"),
            pytest.raises(PermissionDenied),
        ):
            await MemoryService.list_thread_memories(str(thread.id), user=None)


# ============================================================================
# MemoryService -- thread file upload gate
# ============================================================================


# transaction=True so the AgentSettings row is flushed afterwards.
@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
class TestUploadThreadFileRespectsAgentFileUpload:
    async def _thread_for(self, *, file_upload):
        from uuid import uuid4

        from django.contrib.auth import get_user_model
        from django_ai_sdk.agents.models import AgentSettings, AgentUser
        from django_ai_sdk.conversation.models import Thread

        unique = str(uuid4())
        user = await get_user_model().objects.acreate_user(email=f"{unique}@example.com", password="x")
        config = await AgentSettings.objects.acreate(
            name="Uploader", slug=unique, agent="test", file_upload=file_upload
        )
        await AgentUser.objects.acreate(agent=config, user=user)
        thread = await Thread.objects.acreate(
            user=user, metadata={"agent_id": str(config.id)}
        )
        return thread, user

    async def test_an_agent_that_does_not_accept_files_rejects_the_upload(self):
        from django.core.files.base import ContentFile
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.permissions import PermissionDenied

        thread, user = await self._thread_for(file_upload=False)

        with pytest.raises(PermissionDenied):
            await MemoryService.upload_thread_file(
                str(thread.id), ContentFile(b"hello", name="a.txt"), user=user
            )

    async def test_an_agent_that_accepts_files_gets_past_the_gate(self):
        from django.core.files.base import ContentFile
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.permissions import PermissionDenied

        thread, user = await self._thread_for(file_upload=True)

        with patch.object(
            MemoryService, "get_or_create_thread_file_memory", AsyncMock(side_effect=RuntimeError)
        ):
            with pytest.raises(RuntimeError):
                await MemoryService.upload_thread_file(
                    str(thread.id), ContentFile(b"hello", name="a.txt"), user=user
                )



@pytest.mark.django_db
@pytest.mark.asyncio
class TestThreadFileMemory:
    async def test_upload_that_loses_the_race_uses_the_winning_memory(self):
        # Parallel uploads each read the thread before any of them claims it.
        # Replay that: a second call sees a stale thread without file_memory.
        from uuid import uuid4

        from django.contrib.auth import get_user_model
        from django_ai_sdk.conversation.models import Thread
        from django_ai_sdk.memories import services
        from django_ai_sdk.memories.models import Memory, ThreadMemory
        from django_ai_sdk.memories.services import MemoryService

        user = await get_user_model().objects.acreate_user(
            email=f"{uuid4()}@example.com", password="x"
        )
        thread = await Thread.objects.acreate(user=user)
        stale = await Thread.objects.select_related("file_memory").aget(id=thread.id)

        winner = await MemoryService.get_or_create_thread_file_memory(str(thread.id))
        with patch.object(services, "_aget_or_not_found", AsyncMock(return_value=stale)):
            loser = await MemoryService.get_or_create_thread_file_memory(str(thread.id))

        assert loser.id == winner.id
        assert await Memory.objects.filter(name=f"thread_files_{thread.id}").acount() == 1
        assert await ThreadMemory.objects.filter(thread=thread).acount() == 1

    async def test_existing_file_memory_is_reused(self):
        from uuid import uuid4

        from django.contrib.auth import get_user_model
        from django_ai_sdk.conversation.models import Thread
        from django_ai_sdk.memories.models import Memory
        from django_ai_sdk.memories.services import MemoryService

        user = await get_user_model().objects.acreate_user(
            email=f"{uuid4()}@example.com", password="x"
        )
        thread = await Thread.objects.acreate(user=user)

        first = await MemoryService.get_or_create_thread_file_memory(str(thread.id))
        second = await MemoryService.get_or_create_thread_file_memory(str(thread.id))

        assert first.id == second.id
        assert await Memory.objects.filter(name=f"thread_files_{thread.id}").acount() == 1


# ============================================================================
# MemoryService — a thread's memories belong to the thread's owner
# ============================================================================


@pytest.mark.django_db
@pytest.mark.asyncio
class TestThreadMemoryOwnership:
    """A public memory is readable by anyone; someone else's thread is not."""

    async def _setup(self):
        from django_ai_sdk.conversation.models import Thread
        from django_ai_sdk.memories.models import Memory, ThreadMemory
        from tests.factories.db import UserFactory

        owner = await UserFactory.acreate()
        stranger = await UserFactory.acreate()
        thread = await Thread.objects.acreate(user=owner)
        linked = await Memory.objects.acreate(name="linked", is_public=True)
        await ThreadMemory.objects.acreate(thread=thread, memory=linked, active=True)
        public = await Memory.objects.acreate(name="public", is_public=True)
        return owner, stranger, thread, linked, public

    @pytest.mark.parametrize(
        "call",
        [
            lambda s, t, linked, public, u: s.link_memory_to_thread(public, t, user=u),
            lambda s, t, linked, public, u: s.unlink_memory_from_thread(linked, t, user=u),
            lambda s, t, linked, public, u: s.bulk_connect_memories(t, [public], user=u),
            lambda s, t, linked, public, u: s.toggle_memory_active(t, linked, False, user=u),
            lambda s, t, linked, public, u: s.disconnect_memory_from_thread(t, linked, user=u),
            lambda s, t, linked, public, u: s.list_thread_memories(t, user=u),
        ],
        ids=["link", "unlink", "bulk", "toggle", "disconnect", "list"],
    )
    async def test_a_stranger_cannot_touch_someone_elses_thread(self, call):
        from django_ai_sdk.memories.models import ThreadMemory
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.permissions import PermissionDenied

        owner, stranger, thread, linked, public = await self._setup()

        with pytest.raises(PermissionDenied):
            await call(MemoryService, str(thread.id), str(linked.id), str(public.id), stranger)

        active = [
            str(m)
            async for m in ThreadMemory.objects.filter(thread=thread, active=True).values_list(
                "memory_id", flat=True
            )
        ]
        assert active == [str(linked.id)]

    async def test_the_owner_can_link_a_public_memory(self):
        from django_ai_sdk.memories.models import ThreadMemory
        from django_ai_sdk.memories.services import MemoryService

        owner, stranger, thread, linked, public = await self._setup()

        await MemoryService.link_memory_to_thread(str(public.id), str(thread.id), user=owner)

        assert await ThreadMemory.objects.filter(thread=thread, memory=public).aexists()

    async def test_the_thread_file_memory_is_private_to_the_owner(self):
        from django_ai_sdk.conversation.models import Thread
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.permissions import Operation
        from tests.factories.db import UserFactory

        owner = await UserFactory.acreate()
        stranger = await UserFactory.acreate()
        thread = await Thread.objects.acreate(user=owner)

        memory = await MemoryService.get_or_create_thread_file_memory(str(thread.id))

        for operation in (Operation.VIEW_MEMORY, Operation.VIEW_DOCUMENT, Operation.LIST_DOCUMENTS):
            assert await MemoryService.has_perms(owner, operation, memory, raise_on_deny=False)
            assert not await MemoryService.has_perms(
                stranger, operation, memory, raise_on_deny=False
            )


# Same flush reason as TestUploadThreadFileRespectsAgentFileUpload.
@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
class TestThreadFilesBelongToTheThreadOwner:
    """A public agent lets anyone chat with it, not into someone else's thread."""

    async def _setup(self):
        from uuid import uuid4

        from django_ai_sdk.agents.models import AgentSettings
        from django_ai_sdk.conversation.models import Thread
        from tests.factories.db import UserFactory

        owner = await UserFactory.acreate()
        stranger = await UserFactory.acreate()
        config = await AgentSettings.objects.acreate(
            name="Public", slug=str(uuid4()), agent="test", file_upload=True, is_public=True
        )
        thread = await Thread.objects.acreate(user=owner, metadata={"agent_id": str(config.id)})
        return owner, stranger, thread

    @pytest.mark.parametrize(
        "call",
        [
            lambda s, t, u: s.upload_thread_file(t, ContentFile(b"x", name="a.txt"), user=u),
            lambda s, t, u: s.list_thread_files(t, user=u),
            lambda s, t, u: s.delete_thread_file(t, str(uuid4()), user=u),
        ],
        ids=["upload", "list", "delete"],
    )
    async def test_a_stranger_is_refused(self, call):
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.permissions import PermissionDenied

        owner, stranger, thread = await self._setup()

        with pytest.raises(PermissionDenied):
            await call(MemoryService, str(thread.id), stranger)

    async def test_the_owner_lists_their_files(self):
        from django_ai_sdk.memories.services import MemoryService

        owner, stranger, thread = await self._setup()

        assert await MemoryService.list_thread_files(str(thread.id), user=owner) == []
