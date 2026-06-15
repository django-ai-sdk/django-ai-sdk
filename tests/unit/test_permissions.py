"""
Unit tests for the permission system.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from django.test.utils import override_settings




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
        from django_ai_sdk.permissions import IsAdminUser, Operation, check_permissions

        user = MagicMock(is_staff=True)
        await check_permissions(user, Operation.CHAT, [IsAdminUser])

    async def test_is_admin_allows_superuser(self):
        from django_ai_sdk.permissions import IsAdminUser, Operation, check_permissions

        user = MagicMock(is_superuser=True)
        await check_permissions(user, Operation.CHAT, [IsAdminUser])

    async def test_is_admin_denies_regular_user(self):
        from django_ai_sdk.permissions import IsAdminUser, Operation, PermissionDenied, check_permissions

        user = MagicMock(is_staff=False, is_superuser=False)
        with pytest.raises(PermissionDenied):
            await check_permissions(user, Operation.CHAT, [IsAdminUser])

    async def test_is_owner_grants_when_user_id_matches(self):
        from django_ai_sdk.permissions import IsOwner, Operation, check_object_permissions

        obj = MagicMock(user_id="user-1")
        user = MagicMock(pk="user-1")
        await check_object_permissions(user, Operation.DELETE_THREAD, obj, [IsOwner])

    async def test_is_owner_denies_when_user_id_mismatch(self):
        from django_ai_sdk.permissions import IsOwner, Operation, PermissionDenied, check_object_permissions

        obj = MagicMock(user_id="owner-1")
        user = MagicMock(pk="user-1")
        with pytest.raises(PermissionDenied):
            await check_object_permissions(user, Operation.DELETE_THREAD, obj, [IsOwner])

    async def test_is_owner_denies_none_user(self):
        from django_ai_sdk.permissions import IsOwner, Operation, PermissionDenied, check_object_permissions

        obj = MagicMock(user_id="user-1")
        with pytest.raises(PermissionDenied):
            await check_object_permissions(None, Operation.VIEW_THREAD, obj, [IsOwner])

    async def test_is_owner_allows_when_no_obj_user_id(self):
        from django_ai_sdk.permissions import IsOwner, Operation, PermissionDenied, check_permissions

        user = MagicMock(pk="user-1")
        await check_permissions(user, Operation.CREATE_THREAD, [IsOwner])

    async def test_is_owner_grants_all_for_memory_like_object(self):
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

    async def test_view_assistant_allows_with_allow_all(self):
        from django_ai_sdk.permissions import AllowAll, Operation, check_permissions

        await check_permissions(None, Operation.VIEW_ASSISTANT, [AllowAll])

    async def test_view_assistant_denies_anonymous_with_is_authenticated(self):
        from django_ai_sdk.permissions import IsAuthenticated, Operation, PermissionDenied, check_permissions

        with pytest.raises(PermissionDenied):
            await check_permissions(None, Operation.VIEW_ASSISTANT, [IsAuthenticated])

    async def test_view_assistant_denies_regular_user_with_is_admin(self):
        from django_ai_sdk.permissions import IsAdminUser, Operation, PermissionDenied, check_permissions

        user = MagicMock(is_staff=False, is_superuser=False)
        with pytest.raises(PermissionDenied):
            await check_permissions(user, Operation.VIEW_ASSISTANT, [IsAdminUser])

    async def test_view_assistant_allows_admin_with_is_admin(self):
        from django_ai_sdk.permissions import IsAdminUser, Operation, check_permissions

        user = MagicMock(is_staff=True)
        await check_permissions(user, Operation.VIEW_ASSISTANT, [IsAdminUser])

    # --- get_default_permissions ---

    async def test_get_default_permissions_falls_back_to_allow_all(self):
        from django_ai_sdk.permissions import AllowAll, get_default_permissions

        get_default_permissions.cache_clear()
        result = get_default_permissions()
        assert result == [AllowAll]

    async def test_get_default_permissions_from_setting_single(self):
        from django_ai_sdk.permissions import DenyAll, get_default_permissions

        with override_settings(AI_SDK_DEFAULT_PERMISSIONS=["django_ai_sdk.permissions.DenyAll"]):
            get_default_permissions.cache_clear()
            result = get_default_permissions()
            assert result == [DenyAll]
        get_default_permissions.cache_clear()

    async def test_get_default_permissions_from_setting_multiple(self):
        from django_ai_sdk.permissions import DenyAll, IsAuthenticated, get_default_permissions

        with override_settings(
            AI_SDK_DEFAULT_PERMISSIONS=[
                "django_ai_sdk.permissions.DenyAll",
                "django_ai_sdk.permissions.IsAuthenticated",
            ]
        ):
            get_default_permissions.cache_clear()
            result = get_default_permissions()
            assert result == [DenyAll, IsAuthenticated]
        get_default_permissions.cache_clear()

    async def test_get_default_permissions_used_as_fallback_in_assistant_permissions(self):
        from django_ai_sdk.assistants.services import AssistantService
        from django_ai_sdk.permissions import AllowAll, DenyAll, get_default_permissions

        get_default_permissions.cache_clear()
        with override_settings(AI_SDK_DEFAULT_PERMISSIONS=["django_ai_sdk.permissions.DenyAll"]):
            get_default_permissions.cache_clear()
            reg = MagicMock()
            assistant_a = MagicMock(name="a", id="a")
            del assistant_a.permissions
            assistant_b = MagicMock(name="b", id="b", permissions=[AllowAll])
            reg.visible.return_value = {"a": assistant_a, "b": assistant_b}
            reg.get.side_effect = lambda id: reg.visible.return_value.get(id)
            with patch("django_ai_sdk.assistants.services.registry", reg):
                summaries = await AssistantService.list_assistants(None)
                assert len(summaries) == 1
                assert summaries[0]["id"] == "b"

        get_default_permissions.cache_clear()


@pytest.mark.django_db
@pytest.mark.asyncio
class TestMemoryDefaultPermission:
    """Tests for MemoryDefaultPermission three-tier access model."""

    async def _make_owner(self, user, can_manage=False):
        """Helper to create a memory owner."""
        from django_ai_sdk.memories.models import MemoryOwner

        # Need real Memory object with is_public
        from django_ai_sdk.memories.models import Memory

        memory = Memory(name="Test", is_public=False)
        await memory.asave()
        owner = MemoryOwner(user=user, memory=memory, can_manage=can_manage)
        await owner.asave()
        return memory

    async def _make_public_memory(self):
        from django_ai_sdk.memories.models import Memory

        memory = Memory(name="Public Mem", is_public=True)
        await memory.asave()
        return memory

    async def _make_private_memory(self):
        from django_ai_sdk.memories.models import Memory

        memory = Memory(name="Private Mem", is_public=False)
        await memory.asave()
        return memory

    async def test_manager_can_do_anything(self):
        from django_ai_sdk.permissions import (
            MemoryDefaultPermission,
            Operation,
            check_object_permissions,
        )
        from tests.factories.db import UserFactory

        manager = await UserFactory.acreate()
        memory = await self._make_owner(manager, can_manage=True)

        for op in Operation:
            await check_object_permissions(
                manager, op, memory, [MemoryDefaultPermission]
            )

    async def test_contributor_cannot_manage(self):
        from django_ai_sdk.permissions import (
            MemoryDefaultPermission,
            Operation,
            PermissionDenied,
            check_object_permissions,
        )
        from tests.factories.db import UserFactory

        contributor = await UserFactory.acreate()
        memory = await self._make_owner(contributor, can_manage=False)

        for op in MemoryDefaultPermission.MANAGER:
            with pytest.raises(PermissionDenied):
                await check_object_permissions(
                    contributor, op, memory, [MemoryDefaultPermission]
                )

    async def test_contributor_can_read_and_write(self):
        from django_ai_sdk.permissions import (
            MemoryDefaultPermission,
            Operation,
            check_object_permissions,
        )
        from tests.factories.db import UserFactory

        contributor = await UserFactory.acreate()
        memory = await self._make_owner(contributor, can_manage=False)

        allowed_ops = MemoryDefaultPermission.READ | MemoryDefaultPermission.WRITE
        for op in allowed_ops:
            await check_object_permissions(
                contributor, op, memory, [MemoryDefaultPermission]
            )

    async def test_stranger_cannot_access_private_memory(self):
        from django_ai_sdk.permissions import (
            MemoryDefaultPermission,
            Operation,
            PermissionDenied,
            check_object_permissions,
        )
        from tests.factories.db import UserFactory

        stranger = await UserFactory.acreate()
        memory = await self._make_private_memory()

        for op in Operation:
            with pytest.raises(PermissionDenied):
                await check_object_permissions(
                    stranger, op, memory, [MemoryDefaultPermission]
                )

    async def test_stranger_can_read_public_memory(self):
        from django_ai_sdk.permissions import (
            MemoryDefaultPermission,
            Operation,
            PermissionDenied,
            check_object_permissions,
        )
        from tests.factories.db import UserFactory

        stranger = await UserFactory.acreate()
        memory = await self._make_public_memory()

        for op in MemoryDefaultPermission.READ:
            await check_object_permissions(
                stranger, op, memory, [MemoryDefaultPermission]
            )

        for op in MemoryDefaultPermission.WRITE | MemoryDefaultPermission.MANAGER:
            with pytest.raises(PermissionDenied):
                await check_object_permissions(
                    stranger, op, memory, [MemoryDefaultPermission]
                )

    async def test_anonymous_cannot_access_public_memory(self):
        from django_ai_sdk.permissions import (
            MemoryDefaultPermission,
            Operation,
            PermissionDenied,
            check_permissions,
        )

        with pytest.raises(PermissionDenied):
            await check_permissions(
                None, Operation.VIEW_MEMORY, [MemoryDefaultPermission]
            )

    async def test_has_permission_returns_true_for_authenticated(self):
        from django_ai_sdk.permissions import MemoryDefaultPermission, Operation, check_permissions

        from django.contrib.auth.models import User

        authenticated = MagicMock(is_authenticated=True)
        await check_permissions(
            authenticated, Operation.VIEW_MEMORY, [MemoryDefaultPermission]
        )

    async def test_has_permission_denies_anonymous(self):
        from django_ai_sdk.permissions import (
            MemoryDefaultPermission,
            Operation,
            PermissionDenied,
            check_permissions,
        )

        with pytest.raises(PermissionDenied):
            await check_permissions(
                None, Operation.VIEW_MEMORY, [MemoryDefaultPermission]
            )

    async def test_stranger_cannot_write_public_memory(self):
        from django_ai_sdk.permissions import (
            MemoryDefaultPermission,
            Operation,
            PermissionDenied,
            check_object_permissions,
        )
        from tests.factories.db import UserFactory

        stranger = await UserFactory.acreate()
        memory = await self._make_public_memory()

        for op in MemoryDefaultPermission.WRITE:
            with pytest.raises(PermissionDenied):
                await check_object_permissions(
                    stranger, op, memory, [MemoryDefaultPermission]
                )


@pytest.mark.asyncio
class TestAssistantServicePermissions:
    """Permission checks in AssistantService (list_assistants, get_assistant_info)."""

    async def test_list_assistants_filters_by_permission(self):
        from django_ai_sdk.assistants.services import AssistantService
        from django_ai_sdk.permissions import AllowAll, DenyAll

        reg = MagicMock()
        allow_assistant = MagicMock(
            name="allow", id="allow-id", permissions=[AllowAll]
        )
        deny_assistant = MagicMock(name="deny", id="deny-id", permissions=[DenyAll])

        reg.visible.return_value = {"allow": allow_assistant, "deny": deny_assistant}
        reg.get.side_effect = lambda id: reg.visible.return_value.get(id)

        with patch("django_ai_sdk.assistants.services.registry", reg):
            summaries = await AssistantService.list_assistants(None)
            assert len(summaries) == 1
            assert summaries[0]["id"] == "allow"

    async def test_get_assistant_info_allows_with_view_permission(self):
        from django_ai_sdk.assistants.services import AssistantService
        from django_ai_sdk.permissions import AllowAll

        reg = MagicMock()
        assistant = MagicMock(name="test", id="test-id", permissions=[AllowAll])
        assistant.info.return_value = {"id": "test-id", "name": "Test"}

        reg.get.return_value = assistant
        with patch("django_ai_sdk.assistants.services.registry", reg):
            info = await AssistantService.get_assistant_info("test-id")
            assert info["id"] == "test-id"

    async def test_get_assistant_info_denies_without_permission(self):
        from django_ai_sdk.assistants.services import AssistantService
        from django_ai_sdk.permissions import DenyAll, PermissionDenied

        reg = MagicMock()
        assistant = MagicMock(name="test", id="test-id", permissions=[DenyAll])

        reg.get.return_value = assistant
        with patch("django_ai_sdk.assistants.services.registry", reg):
            with pytest.raises(PermissionDenied):
                await AssistantService.get_assistant_info("test-id")

    async def test_get_assistant_info_raises_on_unknown(self):
        from django_ai_sdk.assistants.services import AssistantService

        reg = MagicMock()
        reg.get.return_value = None
        with patch("django_ai_sdk.assistants.services.registry", reg):
            with pytest.raises(ValueError, match="not found"):
                await AssistantService.get_assistant_info("nonexistent")
