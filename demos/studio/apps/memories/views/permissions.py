from __future__ import annotations

from typing import TYPE_CHECKING

from django_ai_sdk.memories.services import MemoryService
from django_ai_sdk.permissions import ObjectPermissions

if TYPE_CHECKING:
    from django_ai_sdk.types import UserType


async def memory_permissions(user: UserType, memory_id: str) -> ObjectPermissions:
    from django.core.exceptions import ValidationError
    from django_ai_sdk.memories.models import Memory

    try:
        memory = await Memory.objects.aget(id=memory_id)
    except (Memory.DoesNotExist, ValidationError):
        return ObjectPermissions()
    raw = await MemoryService.get_object_permissions_map(user, memory)
    return ObjectPermissions(
        can_read=raw.get("read", False),
        can_write=raw.get("write", False),
        can_manage=raw.get("manage", False),
    )
