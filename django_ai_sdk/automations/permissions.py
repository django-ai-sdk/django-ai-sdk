"""Default permissions for the `automations` domain.

Any authenticated user may view automations. Managing one (turning it on or off,
running it now) needs the `MANAGE_PERMISSION` model permission, because both act as
the automation's `run_as` account rather than the caller. `is_staff` grants nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django_ai_sdk.permissions import BasePermission, Operation

if TYPE_CHECKING:
    from django_ai_sdk.types import UserType


class AutomationDefaultPermission(BasePermission):
    MANAGE_PERMISSION = "django_ai_sdk_automations.change_automationstate"

    async def has_permission(self, user: UserType, operation: Operation, **kwargs: Any) -> bool:
        if user is None or not user.is_authenticated:
            return False
        if operation == Operation.VIEW_AUTOMATION:
            return True
        if operation == Operation.MANAGE_AUTOMATION:
            # A user model without PermissionsMixin has no model permissions.
            ahas_perm = getattr(user, "ahas_perm", None)
            return ahas_perm is not None and bool(await ahas_perm(self.MANAGE_PERMISSION))
        return False


__all__ = ["AutomationDefaultPermission"]
