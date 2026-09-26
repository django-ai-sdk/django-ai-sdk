"""Per-object permission flags (can_read/can_write/can_manage) for API responses.

Framework-neutral: the contrib Ninja and DRF layers attach these to thread, agent and
memory payloads so a client knows which actions to offer. A missing or unreadable
object yields all-False rather than an error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django.core.exceptions import ValidationError

from django_ai_sdk.permissions import ObjectPermissions, PermissionDenied

if TYPE_CHECKING:
    from django_ai_sdk.types import UserType


def _flags(raw: dict[str, Any]) -> ObjectPermissions:
    return ObjectPermissions(
        can_read=raw.get("read", False),
        can_write=raw.get("write", False),
        can_manage=raw.get("manage", False),
    )


async def thread_permissions(user: UserType, thread_id: str) -> ObjectPermissions:
    from django_ai_sdk.storage.services import ThreadService

    try:
        thread = await ThreadService.get_thread(thread_id, user=user)
    except PermissionDenied:
        return ObjectPermissions()
    if thread is None:
        return ObjectPermissions()
    return _flags(await ThreadService.get_object_permissions_map(user, thread))


async def agent_permissions(user: UserType, agent_id: str) -> ObjectPermissions:
    from django_ai_sdk.agents.models import AgentSettings
    from django_ai_sdk.agents.services import AgentService

    try:
        config = await AgentSettings.objects.aget(slug=agent_id)
    except AgentSettings.DoesNotExist:
        try:
            config = await AgentSettings.objects.aget(id=agent_id)
        except (AgentSettings.DoesNotExist, ValueError, ValidationError):
            return ObjectPermissions()
    return _flags(await AgentService.get_object_permissions_map(user, config))


async def memory_permissions(user: UserType, memory_id: str) -> ObjectPermissions:
    from django_ai_sdk.memories.models import Memory
    from django_ai_sdk.memories.services import MemoryService

    try:
        memory = await Memory.objects.aget(id=memory_id)
    except (Memory.DoesNotExist, ValidationError):
        return ObjectPermissions()
    return _flags(await MemoryService.get_object_permissions_map(user, memory))
