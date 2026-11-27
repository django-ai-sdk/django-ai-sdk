from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django_ai_sdk.permissions import Operation, ThreadDefaultPermission

if TYPE_CHECKING:
    from django_ai_sdk.storage.schemas import ThreadInfo
    from django_ai_sdk.types import UserType


class DemoThreadPermission(ThreadDefaultPermission):
    async def has_object_permission(
        self,
        user: UserType,
        operation: Operation,
        obj: ThreadInfo,
        **kwargs: Any,
    ) -> bool:
        if user is None or not user.is_authenticated:
            return False

        return obj.user_id is not None and str(obj.user_id) == str(getattr(user, "pk", ""))
