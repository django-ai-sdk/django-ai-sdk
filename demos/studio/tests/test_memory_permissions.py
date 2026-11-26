from __future__ import annotations

import pytest
from asgiref.sync import async_to_sync
from django.contrib.auth.models import AnonymousUser
from django_ai_sdk.memories.models import Memory
from django_ai_sdk.permissions import Operation

from apps.memories.permissions import AllowAnonymousMemoryPermission


@pytest.mark.django_db
def test_anonymous_users_only_read_public_memories() -> None:
    memory = Memory.objects.create(name="Public", is_public=True)
    check = async_to_sync(AllowAnonymousMemoryPermission().has_object_permission)

    assert check(AnonymousUser(), Operation.VIEW_MEMORY, memory)
    assert not check(AnonymousUser(), Operation.UPLOAD_DOCUMENT, memory)
