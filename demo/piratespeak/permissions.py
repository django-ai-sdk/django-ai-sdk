from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django_ai_sdk.permissions import BasePermission, Operation

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser
    from django_ai_sdk.memories.models import Memory


class AllowAnonymousMemoryPermission(BasePermission):
    """Permission class that allows anonymous users read-only access to public memories.

    - Anonymous users: can view/list public memories and documents (read-only).
    - Authenticated users: full access to their own memories + public memories.
    - Manager operations (update/delete memory) require ownership with can_manage=True.
    """

    READ: frozenset[Operation] = frozenset(
        {
            Operation.VIEW_MEMORY,
            Operation.VIEW_DOCUMENT,
            Operation.LIST_DOCUMENTS,
            Operation.LIST_THREAD_MEMORIES,
            Operation.VIEW_FILE,
            Operation.UPLOAD_FILE,
            Operation.DELETE_FILE,
            Operation.UPLOAD_DOCUMENT,
            Operation.DELETE_DOCUMENT,
            Operation.LINK_MEMORY,
            Operation.UNLINK_MEMORY,
        }
    )
    WRITE: frozenset[Operation] = frozenset(
        {
            Operation.UPLOAD_DOCUMENT,
            Operation.DELETE_DOCUMENT,
        }
    )
    MANAGER: frozenset[Operation] = frozenset(
        {
            Operation.UPDATE_MEMORY,
            Operation.DELETE_MEMORY,
        }
    )

    async def has_permission(
        self, user: AbstractBaseUser | AnonymousUser | None, operation: Operation, **kwargs: Any
    ) -> bool:
        # Anonymous users can only read public memories
        if user is None or not user.is_authenticated:
            return operation in self.READ
        return True

    async def has_object_permission(
        self,
        user: AbstractBaseUser | AnonymousUser | None,
        operation: Operation,
        obj: Memory,
        **kwargs: Any,
    ) -> bool:
        # Anonymous users can only read public memories
        if user is None or not user.is_authenticated:
            return operation in self.READ and obj.is_public

        # Check memory user permissions for authenticated users
        ownership = await obj.memory_users.filter(user=user).afirst()
        if ownership is None:
            # Not a memory user — only allow read on public memories
            if operation in self.READ and obj.is_public:
                return True
            return False

        # Memory user: manager ops require can_manage=True
        if operation in self.MANAGER:
            return ownership.can_manage
        return True
