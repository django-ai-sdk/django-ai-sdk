from __future__ import annotations

from typing import TYPE_CHECKING

from django_ai_sdk.assistants.services import AssistantService
from django_ai_sdk.permissions import ObjectPermissions
from django_ai_sdk.storage.services import ThreadService

if TYPE_CHECKING:
    from django_ai_sdk.types import UserType


async def thread_permissions(user: UserType, thread_id: str) -> ObjectPermissions:
    from django_ai_sdk.permissions import PermissionDenied as PermissionDeniedError

    try:
        thread = await ThreadService.get_thread(thread_id, user=user)
    except PermissionDeniedError:
        return ObjectPermissions()
    if thread is None:
        return ObjectPermissions()
    raw = await ThreadService.get_object_permissions_map(user, thread)
    return ObjectPermissions(
        can_read=raw.get("read", False),
        can_write=raw.get("write", False),
        can_manage=raw.get("manage", False),
    )


async def assistant_permissions(user: UserType, assistant_id: str) -> ObjectPermissions:
    from django.core.exceptions import ValidationError
    from django_ai_sdk.assistants.models import AssistantSettings

    try:
        config = await AssistantSettings.objects.aget(slug=assistant_id)
    except AssistantSettings.DoesNotExist:
        try:
            config = await AssistantSettings.objects.aget(id=assistant_id)
        except (AssistantSettings.DoesNotExist, ValueError, ValidationError):
            return ObjectPermissions()
    raw = await AssistantService.get_object_permissions_map(user, config)
    return ObjectPermissions(
        can_read=raw.get("read", False),
        can_write=raw.get("write", False),
        can_manage=raw.get("manage", False),
    )
