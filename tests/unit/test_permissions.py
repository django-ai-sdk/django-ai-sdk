"""
Unit tests for the permission system.
"""

from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from django.test.utils import override_settings




@pytest.mark.django_db
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
        memory = MagicMock(spec=["user_id"], user_id="user-1")
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

    async def test_view_agent_allows_with_allow_all(self):
        from django_ai_sdk.permissions import AllowAll, Operation, check_permissions

        await check_permissions(None, Operation.VIEW_AGENT, [AllowAll])

    async def test_view_agent_denies_anonymous_with_is_authenticated(self):
        from django_ai_sdk.permissions import IsAuthenticated, Operation, PermissionDenied, check_permissions

        with pytest.raises(PermissionDenied):
            await check_permissions(None, Operation.VIEW_AGENT, [IsAuthenticated])

    async def test_view_agent_denies_regular_user_with_is_admin(self):
        from django_ai_sdk.permissions import IsAdminUser, Operation, PermissionDenied, check_permissions

        user = MagicMock(is_staff=False, is_superuser=False)
        with pytest.raises(PermissionDenied):
            await check_permissions(user, Operation.VIEW_AGENT, [IsAdminUser])

    async def test_view_agent_allows_admin_with_is_admin(self):
        from django_ai_sdk.permissions import IsAdminUser, Operation, check_permissions

        user = MagicMock(is_staff=True)
        await check_permissions(user, Operation.VIEW_AGENT, [IsAdminUser])

    # --- get_domain_permissions ---

    async def test_get_domain_permissions_falls_back_to_default(self):
        from django_ai_sdk.permissions import AgentDefaultPermission, get_domain_permissions, PermissionDomain

        get_domain_permissions.cache_clear()
        result = get_domain_permissions(PermissionDomain.AGENT)
        assert result == [AgentDefaultPermission]

    async def test_get_domain_permissions_from_setting_single(self):
        from django_ai_sdk.permissions import DenyAll, get_domain_permissions, PermissionDomain

        with override_settings(AI_SDK_PERMISSIONS={"agent": ["django_ai_sdk.permissions.DenyAll"]}):
            get_domain_permissions.cache_clear()
            result = get_domain_permissions(PermissionDomain.AGENT)
            assert result == [DenyAll]
        get_domain_permissions.cache_clear()

    async def test_get_domain_permissions_from_setting_multiple(self):
        from django_ai_sdk.permissions import DenyAll, IsAuthenticated, get_domain_permissions, PermissionDomain

        with override_settings(
            AI_SDK_PERMISSIONS={
                "agent": [
                    "django_ai_sdk.permissions.DenyAll",
                    "django_ai_sdk.permissions.IsAuthenticated",
                ]
            }
        ):
            get_domain_permissions.cache_clear()
            result = get_domain_permissions(PermissionDomain.AGENT)
            assert result == [DenyAll, IsAuthenticated]
        get_domain_permissions.cache_clear()

    async def test_get_domain_permissions_used_as_fallback_in_agent_permissions(self):
        from django_ai_sdk.agents.services import AgentService
        from django_ai_sdk.permissions import AllowAll, DenyAll, get_domain_permissions, PermissionDomain

        get_domain_permissions.cache_clear()
        with override_settings(AI_SDK_PERMISSIONS={"agent": ["django_ai_sdk.permissions.DenyAll"]}):
            get_domain_permissions.cache_clear()
            reg = MagicMock()
            agent_a = MagicMock(name="a", id="a")
            del agent_a.permissions
            agent_b = MagicMock(name="b", id="b", permissions=[AllowAll])
            reg.visible.return_value = {"a": agent_a, "b": agent_b}
            reg.get.side_effect = lambda id: reg.visible.return_value.get(id)
            with patch("django_ai_sdk.agents.services.registry", reg):
                summaries = await AgentService.list_agents(None)
                assert len(summaries) == 1
                assert summaries[0]["id"] == "b"

        get_domain_permissions.cache_clear()


@pytest.mark.django_db
@pytest.mark.asyncio
class TestMemoryDefaultPermission:
    """Tests for MemoryDefaultPermission three-tier access model."""

    async def _make_memory_user(self, user, can_manage=False):
        """Helper to create a memory user."""
        from django_ai_sdk.memories.models import MemoryUser

        # Need real Memory object with is_public
        from django_ai_sdk.memories.models import Memory

        memory = Memory(name="Test", is_public=False)
        await memory.asave()
        memory_user = MemoryUser(user=user, memory=memory, can_manage=can_manage)
        await memory_user.asave()
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

    async def _make_memory_group(self, user, can_manage=False, memory=None):
        """Helper to link *user* to *memory* (or a fresh one) via a group."""
        from asgiref.sync import sync_to_async
        from django.contrib.auth.models import Group

        from django_ai_sdk.memories.models import Memory, MemoryGroup

        if memory is None:
            memory = Memory(name="Group Mem", is_public=False)
            await memory.asave()

        group = Group(name=f"Test Group {uuid4()}")
        await group.asave()
        await sync_to_async(user.groups.add)(group)

        memory_group = MemoryGroup(memory=memory, group=group, can_manage=can_manage)
        await memory_group.asave()
        return memory

    async def test_manager_can_do_anything(self):
        from django_ai_sdk.permissions import (
            MemoryDefaultPermission,
            Operation,
            check_object_permissions,
        )
        from tests.factories.db import UserFactory

        manager = await UserFactory.acreate()
        memory = await self._make_memory_user(manager, can_manage=True)

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
        memory = await self._make_memory_user(contributor, can_manage=False)

        for op in MemoryDefaultPermission.MANAGE:
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
        memory = await self._make_memory_user(contributor, can_manage=False)

        allowed_ops = MemoryDefaultPermission.READ | MemoryDefaultPermission.WRITE
        for op in allowed_ops:
            await check_object_permissions(
                contributor, op, memory, [MemoryDefaultPermission]
            )

    async def test_group_manage_grant_overrides_weaker_direct_membership(self):
        """A user's direct (non-manager) membership must not mask a manager
        grant coming from a group they also belong to."""
        from django_ai_sdk.permissions import (
            MemoryDefaultPermission,
            check_object_permissions,
        )
        from tests.factories.db import UserFactory

        user = await UserFactory.acreate()
        memory = await self._make_memory_user(user, can_manage=False)
        await self._make_memory_group(user, can_manage=True, memory=memory)

        for op in MemoryDefaultPermission.MANAGE:
            await check_object_permissions(user, op, memory, [MemoryDefaultPermission])

    async def test_direct_manage_grant_overrides_weaker_group_membership(self):
        """The reverse: a manager-level direct membership must not be masked
        by a weaker group membership on the same memory."""
        from django_ai_sdk.permissions import (
            MemoryDefaultPermission,
            check_object_permissions,
        )
        from tests.factories.db import UserFactory

        user = await UserFactory.acreate()
        memory = await self._make_memory_user(user, can_manage=True)
        await self._make_memory_group(user, can_manage=False, memory=memory)

        for op in MemoryDefaultPermission.MANAGE:
            await check_object_permissions(user, op, memory, [MemoryDefaultPermission])

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

        for op in MemoryDefaultPermission.WRITE | MemoryDefaultPermission.MANAGE:
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


@pytest.mark.django_db
@pytest.mark.asyncio
class TestAgentDefaultPermission:
    """Tests for AgentDefaultPermission three-tier access model."""

    async def _make_agent_user(self, user, can_manage=False):
        """Helper to create an agent with a user entry."""
        from django_ai_sdk.agents.models import AgentSettings, AgentUser

        config = AgentSettings(name="Test Agent", slug=str(uuid4()), agent="test")
        await config.asave()
        agent_user = AgentUser(agent=config, user=user, can_manage=can_manage)
        await agent_user.asave()
        return config

    async def _make_agent_group(self, user, can_manage=False, agent=None):
        """Helper to link *user* to *agent* (or a fresh one) via a group."""
        from asgiref.sync import sync_to_async
        from django.contrib.auth.models import Group

        from django_ai_sdk.agents.models import AgentGroup, AgentSettings

        if agent is None:
            agent = AgentSettings(
                name="Group Agent", slug=str(uuid4()), agent="test"
            )
            await agent.asave()

        group = Group(name=f"Test Group {uuid4()}")
        await group.asave()
        await sync_to_async(user.groups.add)(group)

        agent_group = AgentGroup(agent=agent, group=group, can_manage=can_manage)
        await agent_group.asave()
        return agent

    async def _make_private_agent(self):
        from django_ai_sdk.agents.models import AgentSettings

        config = AgentSettings(name="Private Agent", slug=str(uuid4()), agent="test")
        await config.asave()
        return config

    async def test_manager_can_do_anything(self):
        from django_ai_sdk.permissions import (
            AgentDefaultPermission,
            Operation,
            check_object_permissions,
        )
        from tests.factories.db import UserFactory

        manager = await UserFactory.acreate()
        config = await self._make_agent_user(manager, can_manage=True)

        for op in Operation:
            await check_object_permissions(
                manager, op, config, [AgentDefaultPermission]
            )

    async def test_owner_cannot_manage(self):
        from django_ai_sdk.permissions import (
            AgentDefaultPermission,
            Operation,
            PermissionDenied,
            check_object_permissions,
        )
        from tests.factories.db import UserFactory

        owner = await UserFactory.acreate()
        config = await self._make_agent_user(owner, can_manage=False)

        for op in AgentDefaultPermission.MANAGE:
            with pytest.raises(PermissionDenied):
                await check_object_permissions(
                    owner, op, config, [AgentDefaultPermission]
                )

    async def test_owner_can_view_and_chat(self):
        from django_ai_sdk.permissions import (
            AgentDefaultPermission,
            Operation,
            check_object_permissions,
        )
        from tests.factories.db import UserFactory

        owner = await UserFactory.acreate()
        config = await self._make_agent_user(owner, can_manage=False)

        for op in {Operation.VIEW_AGENT, Operation.CHAT}:
            await check_object_permissions(
                owner, op, config, [AgentDefaultPermission]
            )

    async def test_stranger_cannot_access(self):
        from django_ai_sdk.permissions import (
            AgentDefaultPermission,
            Operation,
            PermissionDenied,
            check_object_permissions,
        )
        from tests.factories.db import UserFactory

        stranger = await UserFactory.acreate()
        config = await self._make_private_agent()

        for op in Operation:
            with pytest.raises(PermissionDenied):
                await check_object_permissions(
                    stranger, op, config, [AgentDefaultPermission]
                )

    async def test_group_member_can_view_and_chat(self):
        from django_ai_sdk.permissions import (
            AgentDefaultPermission,
            Operation,
            check_object_permissions,
        )
        from tests.factories.db import UserFactory

        member = await UserFactory.acreate()
        config = await self._make_agent_group(member, can_manage=False)

        for op in {Operation.VIEW_AGENT, Operation.CHAT}:
            await check_object_permissions(
                member, op, config, [AgentDefaultPermission]
            )

    async def test_group_member_cannot_manage(self):
        from django_ai_sdk.permissions import (
            AgentDefaultPermission,
            Operation,
            PermissionDenied,
            check_object_permissions,
        )
        from tests.factories.db import UserFactory

        member = await UserFactory.acreate()
        config = await self._make_agent_group(member, can_manage=False)

        for op in AgentDefaultPermission.MANAGE:
            with pytest.raises(PermissionDenied):
                await check_object_permissions(
                    member, op, config, [AgentDefaultPermission]
                )

    async def test_group_manager_can_manage(self):
        from django_ai_sdk.permissions import (
            AgentDefaultPermission,
            Operation,
            check_object_permissions,
        )
        from tests.factories.db import UserFactory

        manager = await UserFactory.acreate()
        config = await self._make_agent_group(manager, can_manage=True)

        for op in Operation:
            await check_object_permissions(
                manager, op, config, [AgentDefaultPermission]
            )

    async def test_group_manage_grant_overrides_weaker_direct_membership(self):
        """A user's direct (non-manager) membership must not mask a manager
        grant coming from a group they also belong to."""
        from django_ai_sdk.permissions import (
            AgentDefaultPermission,
            check_object_permissions,
        )
        from tests.factories.db import UserFactory

        user = await UserFactory.acreate()
        config = await self._make_agent_user(user, can_manage=False)
        await self._make_agent_group(user, can_manage=True, agent=config)

        for op in AgentDefaultPermission.MANAGE:
            await check_object_permissions(user, op, config, [AgentDefaultPermission])

    async def test_direct_manage_grant_overrides_weaker_group_membership(self):
        """The reverse: a manager-level direct membership must not be masked
        by a weaker group membership on the same agent."""
        from django_ai_sdk.permissions import (
            AgentDefaultPermission,
            check_object_permissions,
        )
        from tests.factories.db import UserFactory

        user = await UserFactory.acreate()
        config = await self._make_agent_user(user, can_manage=True)
        await self._make_agent_group(user, can_manage=False, agent=config)

        for op in AgentDefaultPermission.MANAGE:
            await check_object_permissions(user, op, config, [AgentDefaultPermission])

    async def test_anonymous_denied(self):
        from django_ai_sdk.permissions import (
            AgentDefaultPermission,
            Operation,
            PermissionDenied,
            check_permissions,
        )

        with pytest.raises(PermissionDenied):
            await check_permissions(
                None, Operation.VIEW_AGENT, [AgentDefaultPermission]
            )

    async def test_authenticated_allowed_at_permission_level(self):
        from django_ai_sdk.permissions import (
            AgentDefaultPermission,
            Operation,
            check_permissions,
        )

        user = MagicMock(is_authenticated=True)
        await check_permissions(
            user, Operation.VIEW_AGENT, [AgentDefaultPermission]
        )

    async def test_non_agent_object_passes_through(self):
        """AgentDefaultPermission should not interfere with non-AgentSettings objects."""
        from django_ai_sdk.permissions import (
            AgentDefaultPermission,
            Operation,
            check_object_permissions,
        )
        from tests.factories.db import UserFactory

        user = await UserFactory.acreate()
        some_obj = MagicMock()

        # Should not raise for any operation on non-AgentSettings objects
        for op in Operation:
            await check_object_permissions(
                user, op, some_obj, [AgentDefaultPermission]
            )


@pytest.mark.django_db
@pytest.mark.asyncio
class TestAgentServicePermissions:
    """Permission checks in AgentService (list_agents, get_agent_info)."""

    async def test_list_agents_filters_by_permission(self):
        from django_ai_sdk.agents.models import AgentSettings
        from django_ai_sdk.agents.services import AgentService
        from django_ai_sdk.permissions import AllowAll, DenyAll

        reg = MagicMock()
        allow_agent = MagicMock(
            name="allow", id="allow-id", permissions=[AllowAll]
        )
        deny_agent = MagicMock(name="deny", id="deny-id", permissions=[DenyAll])

        reg.visible.return_value = {"allow": allow_agent, "deny": deny_agent}
        reg.get.side_effect = lambda id: reg.visible.return_value.get(id)

        # Patch DB query to yield nothing so only registry agents are tested
        with patch("django_ai_sdk.agents.services.registry", reg):
            with patch.object(
                AgentSettings.objects, "filter", return_value=AgentSettings.objects.none()
            ):
                summaries = await AgentService.list_agents(None)
                assert len(summaries) == 1
                assert summaries[0]["id"] == "allow"

    async def test_get_agent_info_allows_with_view_permission(self):
        from django_ai_sdk.agents.services import AgentService
        from django_ai_sdk.permissions import AllowAll

        reg = MagicMock()
        agent = MagicMock(name="test", id="test-id", permissions=[AllowAll])
        agent.info.return_value = {"id": "test-id", "name": "Test"}

        user = MagicMock(is_authenticated=True)
        reg.get.return_value = agent
        with patch("django_ai_sdk.agents.services.registry", reg):
            info = await AgentService.get_agent_info("test-id", user=user)
            assert info["id"] == "test-id"

    async def test_get_agent_info_denies_without_permission(self):
        from django_ai_sdk.agents.services import AgentService
        from django_ai_sdk.permissions import DenyAll, PermissionDenied

        user = MagicMock(is_authenticated=True)
        reg = MagicMock()
        agent = MagicMock(name="test", id="test-id", permissions=[DenyAll])

        reg.get.return_value = agent
        with patch("django_ai_sdk.agents.services.registry", reg):
            with pytest.raises(PermissionDenied):
                await AgentService.get_agent_info("test-id", user=user)

    async def test_get_agent_info_raises_on_unknown(self):
        from django_ai_sdk.agents.services import AgentService

        user = MagicMock(is_authenticated=True)
        reg = MagicMock()
        reg.get.return_value = None
        with patch("django_ai_sdk.agents.services.registry", reg):
            with pytest.raises(ValueError, match="not found"):
                await AgentService.get_agent_info("nonexistent", user=user)


# ============================================================================
# IsOwner — configurable field & null-owner behavior
# ============================================================================


@pytest.mark.asyncio
class TestIsOwner:
    async def test_custom_field_parameter(self):
        """IsOwner(field="owner_id") checks the specified attribute."""
        from django_ai_sdk.permissions import IsOwner, Operation, check_object_permissions

        obj = MagicMock(spec=["owner_id"], owner_id="user-1")
        user = MagicMock(pk="user-1")

        await check_object_permissions(
            user, Operation.VIEW_THREAD, obj, [IsOwner(field="owner_id")]
        )

    async def test_custom_field_denies_mismatch(self):
        """IsOwner with custom field denies on mismatch."""
        from django_ai_sdk.permissions import (
            IsOwner,
            Operation,
            PermissionDenied,
            check_object_permissions,
        )

        obj = MagicMock(spec=["owner_id"], owner_id="owner-1")
        user = MagicMock(pk="user-1")

        with pytest.raises(PermissionDenied):
            await check_object_permissions(
                user, Operation.VIEW_THREAD, obj, [IsOwner(field="owner_id")]
            )

    async def test_null_owner_denies_all(self):
        """obj.user_id = None denies even for authenticated users."""
        from django_ai_sdk.permissions import (
            IsOwner,
            Operation,
            PermissionDenied,
            check_object_permissions,
        )

        obj = MagicMock(user_id=None)
        user = MagicMock(pk="user-1", is_authenticated=True)

        with pytest.raises(PermissionDenied):
            await check_object_permissions(
                user, Operation.VIEW_THREAD, obj, [IsOwner]
            )

    async def test_null_owner_denies_anonymous(self):
        """obj.user_id = None denies anonymous users."""
        from django_ai_sdk.permissions import (
            IsOwner,
            Operation,
            PermissionDenied,
            check_object_permissions,
        )

        obj = MagicMock(user_id=None)

        with pytest.raises(PermissionDenied):
            await check_object_permissions(
                None, Operation.VIEW_THREAD, obj, [IsOwner]
            )

    async def test_no_attr_no_opinion(self):
        """Object without the field has no opinion (allows through)."""
        from django_ai_sdk.permissions import IsOwner, Operation, check_permissions

        obj = MagicMock(spec=[])  # no user_id attribute
        user = MagicMock(pk="user-1")

        await check_permissions(
            user, Operation.CREATE_THREAD, [IsOwner], obj=obj
        )


# ============================================================================
# ObjectPermissions schema and mixin
# ============================================================================


@pytest.mark.asyncio
class TestObjectPermissionsSchema:
    """Tests for the ObjectPermissions schema and ObjectPermsSchema mixin."""

    async def test_default_values_all_false(self):
        from django_ai_sdk.permissions import ObjectPermissions

        perms = ObjectPermissions()
        assert perms.can_read is False
        assert perms.can_write is False
        assert perms.can_manage is False

    async def test_custom_values(self):
        from django_ai_sdk.permissions import ObjectPermissions

        perms = ObjectPermissions(can_read=True, can_write=False, can_manage=True)
        assert perms.can_read is True
        assert perms.can_write is False
        assert perms.can_manage is True

    async def test_serializes_as_dict(self):
        from django_ai_sdk.permissions import ObjectPermissions

        perms = ObjectPermissions(can_read=True, can_write=True, can_manage=False)
        d = perms.model_dump()
        assert d == {"can_read": True, "can_write": True, "can_manage": False}

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
        from tests.factories.db import UserFactory

        return await UserFactory.acreate()

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


@pytest.mark.django_db
@pytest.mark.asyncio
class TestAgentDefaultPermissionCreator:
    """Creator automatically becomes manager via create_runtime_agent."""

    async def _make_user(self):
        from tests.factories.db import UserFactory

        return await UserFactory.acreate()

    async def test_creator_is_manager(self):
        """After create_runtime_agent, creator can manage."""
        from django_ai_sdk.agents.services import AgentService
        from django_ai_sdk.permissions import (
            AgentDefaultPermission,
            Operation,
            check_object_permissions,
        )

        user = await self._make_user()

        config = await AgentService.create_runtime_agent(
            {
                "name": "Creator Test",
                "slug": "creator-test",
                "agent": "",
                "model": "gpt-4o",
            },
            user=user,
        )

        # Creator should be able to update (manager operation)
        await check_object_permissions(
            user, Operation.UPDATE_AGENT, config, [AgentDefaultPermission]
        )

        # Creator should be able to delete (manager operation)
        await check_object_permissions(
            user, Operation.DELETE_AGENT, config, [AgentDefaultPermission]
        )

    async def test_stranger_cannot_manage(self):
        """Non-member cannot perform manager operations."""
        from django_ai_sdk.agents.services import AgentService
        from django_ai_sdk.permissions import (
            AgentDefaultPermission,
            Operation,
            PermissionDenied,
            check_object_permissions,
        )

        creator = await self._make_user()
        stranger = await self._make_user()

        config = await AgentService.create_runtime_agent(
            {
                "name": "Stranger Test",
                "slug": "stranger-test",
                "agent": "",
                "model": "gpt-4o",
            },
            user=creator,
        )

        with pytest.raises(PermissionDenied):
            await check_object_permissions(
                stranger, Operation.UPDATE_AGENT, config, [AgentDefaultPermission]
            )

    async def test_add_user_upsert(self):
        """Adding existing user updates can_manage flag."""
        from django_ai_sdk.agents.models import AgentUser
        from django_ai_sdk.agents.services import AgentService

        creator = await self._make_user()
        member = await self._make_user()

        config = await AgentService.create_runtime_agent(
            {
                "name": "Upsert Test",
                "slug": "upsert-test",
                "agent": "",
                "model": "gpt-4o",
            },
            user=creator,
        )

        # Add as viewer
        await AgentService.add_agent_user(
            str(config.id), str(member.id), can_manage=False, user=creator
        )

        # Re-add as manager (upsert)
        await AgentService.add_agent_user(
            str(config.id), str(member.id), can_manage=True, user=creator
        )

        entry = await AgentUser.objects.aget(agent=config, user=member)
        assert entry.can_manage is True

    async def test_add_group_upsert(self):
        """Adding existing group updates can_manage flag."""
        from django.contrib.auth.models import Group
        from django_ai_sdk.agents.models import AgentGroup
        from django_ai_sdk.agents.services import AgentService

        creator = await self._make_user()
        group = await Group.objects.acreate(name="test-group")

        config = await AgentService.create_runtime_agent(
            {
                "name": "Group Upsert",
                "slug": "group-upsert",
                "agent": "",
                "model": "gpt-4o",
            },
            user=creator,
        )

        # Add as viewer
        await AgentService.add_agent_group(
            str(config.id), group.id, can_manage=False, user=creator
        )

        # Re-add as manager (upsert)
        await AgentService.add_agent_group(
            str(config.id), group.id, can_manage=True, user=creator
        )

        entry = await AgentGroup.objects.aget(agent=config, group=group)
        assert entry.can_manage is True


@pytest.mark.django_db
@pytest.mark.asyncio
class TestAgentUserManagement:
    """Agent owner/manager can add users; strangers and viewers are blocked."""

    async def _make_user(self):
        from tests.factories.db import UserFactory

        return await UserFactory.acreate()

    async def test_manager_can_add_user(self):
        """Agent manager can add other users as viewers."""
        from unittest.mock import patch

        from django_ai_sdk.agents.models import AgentSettings, AgentUser
        from django_ai_sdk.agents.services import AgentService
        from django_ai_sdk.permissions import AgentDefaultPermission

        creator = await self._make_user()
        viewer = await self._make_user()

        config = await AgentService.create_runtime_agent(
            {
                "name": "Add User Test",
                "slug": "add-user-test",
                "agent": "",
                "model": "gpt-4o",
            },
            user=creator,
        )

        with patch(
            "django_ai_sdk.agents.services.get_agent_permissions",
            return_value=[AgentDefaultPermission],
        ):
            result = await AgentService.add_agent_user(
                str(config.id), str(viewer.id), can_manage=False, user=creator
            )

        assert result.can_manage is False
        assert await AgentUser.objects.filter(agent=config, user=viewer).aexists()

    async def test_manager_can_promote_user_to_manager(self):
        """Agent manager can promote viewer to manager."""
        from unittest.mock import patch

        from django_ai_sdk.agents.models import AgentSettings, AgentUser
        from django_ai_sdk.agents.services import AgentService
        from django_ai_sdk.permissions import AgentDefaultPermission

        creator = await self._make_user()
        member = await self._make_user()

        config = await AgentService.create_runtime_agent(
            {
                "name": "Promote Test",
                "slug": "promote-test",
                "agent": "",
                "model": "gpt-4o",
            },
            user=creator,
        )

        with patch(
            "django_ai_sdk.agents.services.get_agent_permissions",
            return_value=[AgentDefaultPermission],
        ):
            await AgentService.add_agent_user(
                str(config.id), str(member.id), can_manage=False, user=creator
            )

            result = await AgentService.update_agent_user(
                str(config.id), str(member.id), can_manage=True, user=creator
            )

        assert result.can_manage is True

    async def test_stranger_cannot_add_user(self):
        """Non-member cannot add users to agent."""
        from unittest.mock import patch

        from django_ai_sdk.agents.models import AgentSettings
        from django_ai_sdk.agents.services import AgentService
        from django_ai_sdk.permissions import AgentDefaultPermission, PermissionDenied

        creator = await self._make_user()
        stranger = await self._make_user()
        viewer = await self._make_user()

        config = await AgentService.create_runtime_agent(
            {
                "name": "Stranger Block",
                "slug": "stranger-block",
                "agent": "",
                "model": "gpt-4o",
            },
            user=creator,
        )

        with patch(
            "django_ai_sdk.agents.services.get_agent_permissions",
            return_value=[AgentDefaultPermission],
        ):
            with pytest.raises(PermissionDenied):
                await AgentService.add_agent_user(
                    str(config.id), str(viewer.id), can_manage=False, user=stranger
                )

    async def test_viewer_cannot_add_user(self):
        """Non-manager member cannot add other users."""
        from unittest.mock import patch

        from django_ai_sdk.agents.models import AgentSettings
        from django_ai_sdk.agents.services import AgentService
        from django_ai_sdk.permissions import AgentDefaultPermission, PermissionDenied

        creator = await self._make_user()
        viewer = await self._make_user()
        new_user = await self._make_user()

        config = await AgentService.create_runtime_agent(
            {
                "name": "Viewer Block",
                "slug": "viewer-block",
                "agent": "",
                "model": "gpt-4o",
            },
            user=creator,
        )

        with patch(
            "django_ai_sdk.agents.services.get_agent_permissions",
            return_value=[AgentDefaultPermission],
        ):
            await AgentService.add_agent_user(
                str(config.id), str(viewer.id), can_manage=False, user=creator
            )

            with pytest.raises(PermissionDenied):
                await AgentService.add_agent_user(
                    str(config.id), str(new_user.id), can_manage=False, user=viewer
                )


@pytest.mark.django_db
@pytest.mark.asyncio
class TestAgentDefaultPermissionUse:
    """Tests for gating *use* of a DB-backed agent by its row."""

    async def _runtime_agent(self, *, is_public=False):
        from django_ai_sdk.agents.models import AgentSettings
        from django_ai_sdk.agents.runtime import RuntimeAgent

        config = AgentSettings(
            name="Private Agent", slug=str(uuid4()), agent="test", is_public=is_public
        )
        await config.asave()
        return RuntimeAgent(config), config

    async def _user(self, username):
        from django.contrib.auth import get_user_model

        return await get_user_model().objects.acreate_user(email=f"{username}@example.com", password="x")

    async def test_a_non_member_may_not_chat_with_a_private_agent(self):
        from django_ai_sdk.permissions import AgentDefaultPermission, Operation

        agent, _ = await self._runtime_agent()
        outsider = await self._user("outsider")

        allowed = await AgentDefaultPermission().has_permission(
            outsider, Operation.CHAT, agent=agent
        )
        assert allowed is False

    async def test_a_member_may_chat_with_a_private_agent(self):
        from django_ai_sdk.agents.models import AgentUser
        from django_ai_sdk.permissions import AgentDefaultPermission, Operation

        agent, config = await self._runtime_agent()
        member = await self._user("member")
        await AgentUser(agent=config, user=member).asave()

        allowed = await AgentDefaultPermission().has_permission(member, Operation.CHAT, agent=agent)
        assert allowed is True

    async def test_a_group_member_may_chat_with_a_private_agent(self):
        from asgiref.sync import sync_to_async
        from django.contrib.auth.models import Group

        from django_ai_sdk.agents.models import AgentGroup
        from django_ai_sdk.permissions import AgentDefaultPermission, Operation

        agent, config = await self._runtime_agent()
        member = await self._user("groupie")
        group = await Group.objects.acreate(name="team")
        await sync_to_async(member.groups.add)(group)
        await AgentGroup(agent=config, group=group).asave()

        allowed = await AgentDefaultPermission().has_permission(member, Operation.CHAT, agent=agent)
        assert allowed is True

    async def test_a_public_agent_admits_anyone(self):
        from django_ai_sdk.permissions import AgentDefaultPermission, Operation

        agent, _ = await self._runtime_agent(is_public=True)
        outsider = await self._user("passerby")

        allowed = await AgentDefaultPermission().has_permission(
            outsider, Operation.CHAT, agent=agent
        )
        assert allowed is True

    async def test_agent_crud_is_not_judged_by_the_use_branch(self):
        from django_ai_sdk.permissions import AgentDefaultPermission, Operation

        agent, _ = await self._runtime_agent()
        outsider = await self._user("crud")

        allowed = await AgentDefaultPermission().has_permission(
            outsider, Operation.VIEW_AGENT, agent=agent
        )
        assert allowed is True

    async def test_a_code_declared_agent_has_no_row_to_gate_on(self):
        from django_ai_sdk.permissions import AgentDefaultPermission, Operation

        outsider = await self._user("registry")
        allowed = await AgentDefaultPermission().has_permission(
            outsider, Operation.CHAT, agent=MagicMock(spec=[], is_runtime=False)
        )
        assert allowed is True

    async def test_a_missing_agent_keyword_denies(self):
        from django_ai_sdk.permissions import AgentDefaultPermission, Operation

        outsider = await self._user("forgetful")
        allowed = await AgentDefaultPermission().has_permission(outsider, Operation.CHAT)
        assert allowed is False


class TestGetAgentPermissionsFallback:
    """An empty list must fall back, not silently switch every check off.

    The three resolvers used to disagree about what [] meant, and the agent one was the
    outlier where it disabled gating.
    """

    def _resolve(self, perms):
        from django_ai_sdk.permissions import get_agent_permissions

        agent = MagicMock()
        agent.permissions = perms
        return get_agent_permissions(agent)

    def test_an_empty_list_falls_back_to_the_domain_default(self):
        from django_ai_sdk.permissions import AgentDefaultPermission

        assert self._resolve([]) == [AgentDefaultPermission]

    def test_none_falls_back_to_the_domain_default(self):
        from django_ai_sdk.permissions import AgentDefaultPermission

        assert self._resolve(None) == [AgentDefaultPermission]

    def test_a_non_empty_list_overrides_the_domain_default(self):
        from django_ai_sdk.permissions import AllowAll

        assert self._resolve([AllowAll]) == [AllowAll]

    def test_the_agent_and_integration_resolvers_agree_on_an_empty_list(self):
        from django_ai_sdk.permissions import (
            PermissionDomain,
            get_agent_permissions,
            get_domain_permissions,
            get_integration_permissions,
        )

        blank = MagicMock()
        blank.permissions = []
        assert get_agent_permissions(blank) == get_domain_permissions(PermissionDomain.AGENT)
        assert get_integration_permissions(blank) == get_domain_permissions(
            PermissionDomain.INTEGRATIONS
        )
