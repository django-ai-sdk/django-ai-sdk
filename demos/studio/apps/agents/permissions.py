from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django_ai_sdk.oversight.permissions import OversightDefaultPermission
from django_ai_sdk.permissions import (
    Operation,
    ThreadDefaultPermission,
    WorkflowDefaultPermission,
)

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


def is_overseer(user: UserType) -> bool:
    return bool(
        user
        and user.is_authenticated
        and (getattr(user, "is_staff", False) or getattr(user, "is_superuser", False))
    )


class StudioOversightPermission(OversightDefaultPermission):
    async def has_permission(self, user: UserType, operation: Operation, **kwargs: Any) -> bool:
        if is_overseer(user):
            return True
        return await super().has_permission(user, operation, **kwargs)


class StudioWorkflowPermission(WorkflowDefaultPermission):
    """Lets an approver decide on their own run. Demo only: drops the SDK's four-eyes rule."""

    async def has_object_permission(
        self,
        user: UserType,
        operation: Operation,
        obj: Any,
        **kwargs: Any,
    ) -> bool:
        if operation is Operation.APPROVE_WORKFLOW:
            return await self.has_permission(user, operation)
        return await super().has_object_permission(user, operation, obj, **kwargs)
