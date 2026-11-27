from __future__ import annotations

from typing import TYPE_CHECKING

from asgiref.sync import async_to_sync
from django.core.exceptions import ValidationError

from django_ai_sdk.permissions import ObjectPermissions, PermissionDenied

if TYPE_CHECKING:
    from django_ai_sdk.types import UserType


async def athread_permissions(user: UserType, thread_id: str) -> ObjectPermissions:
    """All-False for a missing or unreadable object, never an error."""
    from django_ai_sdk.storage.services import ThreadService

    try:
        thread = await ThreadService.get_thread(thread_id, user=user)
    except PermissionDenied:
        return ObjectPermissions()
    if thread is None:
        return ObjectPermissions()
    return await ThreadService.get_object_permissions(user, thread)


async def aagent_permissions(user: UserType, agent_id: str) -> ObjectPermissions:
    """All-False for a missing or unreadable object, never an error."""
    from django_ai_sdk.agents.models import AgentSettings
    from django_ai_sdk.agents.services import AgentService

    try:
        config = await AgentSettings.objects.aget(slug=agent_id)
    except AgentSettings.DoesNotExist:
        try:
            config = await AgentSettings.objects.aget(id=agent_id)
        except (AgentSettings.DoesNotExist, ValueError, ValidationError):
            return ObjectPermissions()
    return await AgentService.get_object_permissions(user, config)


async def amemory_permissions(user: UserType, memory_id: str) -> ObjectPermissions:
    """All-False for a missing or unreadable object, never an error."""
    from django_ai_sdk.memories.models import Memory
    from django_ai_sdk.memories.services import MemoryService

    try:
        memory = await Memory.objects.aget(id=memory_id)
    except (Memory.DoesNotExist, ValidationError):
        return ObjectPermissions()
    return await MemoryService.get_object_permissions(user, memory)


thread_permissions = async_to_sync(athread_permissions)
agent_permissions = async_to_sync(aagent_permissions)
memory_permissions = async_to_sync(amemory_permissions)
