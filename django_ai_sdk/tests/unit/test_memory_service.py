"""
Unit tests for MemoryService permission enforcement.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from django_ai_sdk.tests.factories.schemas import ThreadInfoFactory
from django_ai_sdk.tests.mocks.assistant import create_mock_adapter_class


# ============================================================================
# MemoryService — get_assistant_memories
# ============================================================================


@pytest.mark.django_db
@pytest.mark.asyncio
class TestMemoryServiceGetAssistantMemories:
    async def test_filters_by_slug(self):
        from unittest.mock import patch
        from django_ai_sdk.memories.models import Memory
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.assistants.services import registry

        mem1 = await Memory.objects.acreate(name="Legal Documents")
        await Memory.objects.acreate(name="General Docs")

        mock_assistant = MagicMock()
        mock_assistant.memories = [mem1.slug]

        with patch.object(registry, "get", return_value=mock_assistant):
            result = await MemoryService.get_assistant_memories("test-asst")
            assert result == [str(mem1.id)]

    async def test_returns_empty_list_when_no_memories(self):
        from unittest.mock import patch
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.assistants.services import registry

        mock_assistant = MagicMock()
        mock_assistant.memories = []

        with patch.object(registry, "get", return_value=mock_assistant):
            result = await MemoryService.get_assistant_memories("test-asst")
            assert result == []

    async def test_skips_nonexistent_slugs(self):
        from unittest.mock import patch
        from django_ai_sdk.memories.services import MemoryService
        from django_ai_sdk.assistants.services import registry

        mock_assistant = MagicMock()
        mock_assistant.memories = ["nonexistent-slug"]

        with patch.object(registry, "get", return_value=mock_assistant):
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
        from django_ai_sdk.tests.factories.db import UserFactory

        return await UserFactory.acreate(is_superuser=True)

    async def _get_user(self, pk, username):
        from django_ai_sdk.tests.factories.db import UserFactory

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
        from django_ai_sdk.memories.models import Memory, MemoryOwner
        from django.test.utils import override_settings

        owner = await self._get_user(20, "owner2")

        mem = await Memory.objects.acreate(
            name="to-delete", description="x", is_public=False
        )
        await MemoryOwner.objects.acreate(memory=mem, user=owner, can_manage=True)

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
        from django_ai_sdk.memories.models import Memory, MemoryOwner
        from django_ai_sdk.permissions import PermissionDenied
        from django.test.utils import override_settings

        owner = await self._get_user(30, "owner3")
        stranger = await self._get_user(31, "stranger3")

        mem = await Memory.objects.acreate(
            name="private", description="x", is_public=False
        )
        await MemoryOwner.objects.acreate(memory=mem, user=owner, can_manage=True)

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
        from django_ai_sdk.memories.models import Memory, MemoryOwner
        from django.test.utils import override_settings

        owner = await self._get_user(40, "owner4")
        stranger = await self._get_user(41, "stranger4")

        mem = await Memory.objects.acreate(
            name="public", description="x", is_public=True
        )
        await MemoryOwner.objects.acreate(memory=mem, user=owner, can_manage=True)

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
        from django_ai_sdk.memories.models import Memory, MemoryOwner
        from django_ai_sdk.permissions import PermissionDenied
        from django.test.utils import override_settings

        owner = await self._get_user(50, "owner5")
        stranger = await self._get_user(51, "stranger5")

        mem = await Memory.objects.acreate(
            name="private-no-peek", description="x", is_public=False
        )
        await MemoryOwner.objects.acreate(memory=mem, user=owner, can_manage=True)

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
        from django_ai_sdk.memories.models import Memory, MemoryOwner, ThreadMemory
        from django_ai_sdk.conversation.models import Thread
        from django.test.utils import override_settings
        from django_ai_sdk.assistants.services import registry

        owner = await self._get_user(60, "link-owner")
        thread = await Thread.objects.acreate()

        mem = await Memory.objects.acreate(
            name="link-test", is_public=False
        )
        await MemoryOwner.objects.acreate(memory=mem, user=owner, can_manage=True)

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
        from django_ai_sdk.memories.models import Memory, MemoryOwner, ThreadMemory
        from django_ai_sdk.conversation.models import Thread
        from django.test.utils import override_settings
        from django_ai_sdk.assistants.services import registry

        owner = await self._get_user(61, "unlink-owner")
        thread = await Thread.objects.acreate()

        mem = await Memory.objects.acreate(
            name="unlink-test", is_public=False
        )
        await MemoryOwner.objects.acreate(memory=mem, user=owner, can_manage=True)
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
        from django_ai_sdk.memories.models import Memory, MemoryOwner
        from django_ai_sdk.conversation.models import Thread
        from django_ai_sdk.permissions import PermissionDenied
        from django_ai_sdk.assistants.services import registry
        from django.test.utils import override_settings

        owner = await self._get_user(70, "link-user-req")
        thread = await Thread.objects.acreate()

        mem = await Memory.objects.acreate(
            name="req-link", is_public=False
        )
        await MemoryOwner.objects.acreate(memory=mem, user=owner, can_manage=True)

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
        from django_ai_sdk.memories.models import Memory, MemoryOwner
        from django_ai_sdk.conversation.models import Thread
        from django_ai_sdk.permissions import PermissionDenied
        from django_ai_sdk.assistants.services import registry
        from django.test.utils import override_settings

        owner = await self._get_user(71, "unlink-user-req")
        thread = await Thread.objects.acreate()

        mem = await Memory.objects.acreate(
            name="req-unlink", is_public=False
        )
        await MemoryOwner.objects.acreate(memory=mem, user=owner, can_manage=True)

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
            name="ownerless", description="x", is_public=False
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

        thread_info = ThreadInfoFactory.build(assistant_id="test-assistant", user_id=None)
        mock_adapter_cls = create_mock_adapter_class(get_thread=thread_info)
        mock_storage_adapter_registry.get_all_adapters.return_value = [mock_adapter_cls]

        result = await ThreadService.get_thread("thread-1", user=None)
        assert result is not None
        assert result.assistant_id == "test-assistant"

    async def test_thread_service_storage_for_thread_with_none_user(
        self, mock_assistants_registry
    ):
        from django_ai_sdk.storage.services import ThreadService

        thread_info = ThreadInfoFactory.build(assistant_id="test-assistant", user_id=None)

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

        thread_info = ThreadInfoFactory.build(assistant_id="test-assistant", user_id=None)
        mock_adapter_cls = create_mock_adapter_class(get_thread=thread_info)
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
