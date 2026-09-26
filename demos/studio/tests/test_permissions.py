"""Permission wiring in the studio demo: its response schemas and its per-object
permission calculators (apps.*.views.permissions)."""

import uuid

import pytest
from django.contrib.auth import get_user_model


@pytest.mark.django_db
@pytest.mark.asyncio
class TestMemoryOutResponsePermissions:
    async def test_memory_out_response_has_permissions_field(self):
        from apps.memories.views.ninja import MemoryOutResponse
        from django_ai_sdk.permissions import ObjectPermissions

        instance = MemoryOutResponse(
            id="mem-1",
            name="Test",
            slug="test",
            description="",
            is_public=False,
            document_count=0,
            created_at="2024-01-01",
            updated_at="2024-01-01",
        )
        assert hasattr(instance, "permissions")
        assert isinstance(instance.permissions, ObjectPermissions)
        assert instance.permissions.can_read is False

    async def test_memory_out_response_with_custom_permissions(self):
        from apps.memories.views.ninja import MemoryOutResponse
        from django_ai_sdk.permissions import ObjectPermissions

        perms = ObjectPermissions(can_read=True, can_write=False, can_manage=True)
        instance = MemoryOutResponse(
            id="mem-2",
            name="Test",
            slug="test",
            description="",
            is_public=False,
            document_count=0,
            created_at="2024-01-01",
            updated_at="2024-01-01",
            permissions=perms,
        )
        assert instance.permissions.can_read is True
        assert instance.permissions.can_manage is True

    async def test_multiple_inheritance_with_memory_out(self):
        from apps.memories.views.ninja import MemoryOutResponse
        from django_ai_sdk.permissions import ObjectPermissions

        instance = MemoryOutResponse(
            id="mem-1",
            name="Test",
            slug="test",
            description="",
            is_public=False,
            document_count=0,
            created_at="2024-01-01",
            updated_at="2024-01-01",
            permissions=ObjectPermissions(can_read=True, can_write=True, can_manage=True),
        )
        assert instance.id == "mem-1"
        assert instance.permissions.can_read is True
        assert instance.permissions.can_manage is True


@pytest.mark.django_db
@pytest.mark.asyncio
class TestObjectPermissionsCalculators:
    """Integration tests for memory/thread/agent permission calculators."""

    async def _make_user(self):
        return await get_user_model().objects.acreate(email=f"{uuid.uuid4().hex}@example.com")

    # --- memory_permissions ---

    async def test_memory_permissions_owner_gets_all(self):
        from django_ai_sdk.memories.models import Memory, MemoryUser
        from apps.memories.views.permissions import memory_permissions

        user = await self._make_user()
        memory = await Memory.objects.acreate(name="Owner Mem", is_public=False)
        await MemoryUser.objects.acreate(user=user, memory=memory, can_manage=True)

        perms = await memory_permissions(user, str(memory.id))
        assert perms.can_read is True
        assert perms.can_write is True
        assert perms.can_manage is True

    async def test_memory_permissions_stranger_on_public_only_read(self):
        from django_ai_sdk.memories.models import Memory
        from apps.memories.views.permissions import memory_permissions

        user = await self._make_user()
        memory = await Memory.objects.acreate(name="Public Mem", is_public=True)

        perms = await memory_permissions(user, str(memory.id))
        # Demo's AllowAnonymousMemoryPermission puts UPLOAD_DOCUMENT/DELETE_DOCUMENT
        # in both READ and WRITE frozensets, so authenticated strangers can write
        # to public memories but not manage them.
        assert perms.can_read is True
        assert perms.can_write is True
        assert perms.can_manage is False

    async def test_memory_permissions_stranger_on_private_gets_none(self):
        from django_ai_sdk.memories.models import Memory
        from apps.memories.views.permissions import memory_permissions

        user = await self._make_user()
        memory = await Memory.objects.acreate(name="Private Mem", is_public=False)

        perms = await memory_permissions(user, str(memory.id))
        assert perms.can_read is False
        assert perms.can_write is False
        assert perms.can_manage is False

    async def test_memory_permissions_nonexistent_memory_returns_default(self):
        from apps.memories.views.permissions import memory_permissions

        user = await self._make_user()
        perms = await memory_permissions(user, "nonexistent-id")
        assert perms.can_read is False
        assert perms.can_write is False
        assert perms.can_manage is False

    # --- thread_permissions ---

    async def test_thread_permissions_owner_gets_all(self):
        from uuid import uuid4

        from django_ai_sdk.storage.db import DbStorageAdapter
        from django_ai_sdk.storage.services import ThreadService
        from apps.agents.views.permissions import thread_permissions

        user = await self._make_user()
        thread_id = str(uuid4())
        await DbStorageAdapter.create_thread(
            title="Test Thread",
            metadata={"agent_id": "test"},
            user=user,
            thread_id=thread_id,
        )

        perms = await thread_permissions(user, thread_id)
        assert perms.can_read is True
        assert perms.can_write is True
        assert perms.can_manage is True

    async def test_thread_permissions_stranger_gets_none(self):
        from uuid import uuid4

        from django_ai_sdk.storage.db import DbStorageAdapter
        from apps.agents.views.permissions import thread_permissions

        owner = await self._make_user()
        stranger = await self._make_user()
        thread_id = str(uuid4())
        await DbStorageAdapter.create_thread(
            title="Test Thread",
            metadata={"agent_id": "test"},
            user=owner,
            thread_id=thread_id,
        )

        perms = await thread_permissions(stranger, thread_id)
        assert perms.can_read is False
        assert perms.can_write is False
        assert perms.can_manage is False

    async def test_thread_permissions_nonexistent_returns_default(self):
        from apps.agents.views.permissions import thread_permissions

        user = await self._make_user()
        perms = await thread_permissions(user, "nonexistent-id")
        assert perms.can_read is False
        assert perms.can_write is False
        assert perms.can_manage is False

    # --- agent_permissions ---

    async def test_agent_permissions_owner_gets_all(self):
        from django_ai_sdk.agents.models import AgentSettings, AgentUser
        from apps.agents.views.permissions import agent_permissions

        user = await self._make_user()
        config = await AgentSettings.objects.acreate(
            name="Test", slug="test-slug", agent="test"
        )
        await AgentUser.objects.acreate(agent=config, user=user, can_manage=True)

        perms = await agent_permissions(user, "test-slug")
        assert perms.can_read is True
        assert perms.can_write is True
        assert perms.can_manage is True

    async def test_agent_permissions_stranger_gets_none(self):
        from django_ai_sdk.agents.models import AgentSettings
        from apps.agents.views.permissions import agent_permissions

        stranger = await self._make_user()
        await AgentSettings.objects.acreate(
            name="Private", slug="private-slug", agent="test"
        )

        perms = await agent_permissions(stranger, "private-slug")
        assert perms.can_read is False
        assert perms.can_write is False
        assert perms.can_manage is False

    async def test_agent_permissions_nonexistent_returns_default(self):
        from apps.agents.views.permissions import agent_permissions

        user = await self._make_user()
        perms = await agent_permissions(user, "nonexistent")
        assert perms.can_read is False
        assert perms.can_write is False
        assert perms.can_manage is False

    async def test_agent_permissions_looks_up_by_id_fallback(self):
        from django_ai_sdk.agents.models import AgentSettings, AgentUser
        from apps.agents.views.permissions import agent_permissions

        user = await self._make_user()
        config = await AgentSettings.objects.acreate(
            name="By ID", slug="by-id-slug", agent="test"
        )
        await AgentUser.objects.acreate(agent=config, user=user, can_manage=True)

        perms = await agent_permissions(user, str(config.id))
        assert perms.can_read is True
        assert perms.can_manage is True


# ============================================================================
# AgentDefaultPermission — creator-as-manager, upserts
# ============================================================================
