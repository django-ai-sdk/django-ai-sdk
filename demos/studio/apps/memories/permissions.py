from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django_ai_sdk.permissions import BasePermission, Operation

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser
    from django.db.models import QuerySet
    from django_ai_sdk.memories.models import Memory
    from django_ai_sdk.types import UserType


class AllowAnonymousMemoryPermission(BasePermission):
    READ: frozenset[Operation] = frozenset(
        {
            Operation.VIEW_MEMORY,
            Operation.VIEW_DOCUMENT,
            Operation.LIST_DOCUMENTS,
            Operation.LIST_THREAD_MEMORIES,
            Operation.VIEW_FILE,
        }
    )
    WRITE: frozenset[Operation] = frozenset(
        {
            Operation.UPLOAD_FILE,
            Operation.DELETE_FILE,
            Operation.UPLOAD_DOCUMENT,
            Operation.DELETE_DOCUMENT,
            Operation.LINK_MEMORY,
            Operation.UNLINK_MEMORY,
        }
    )
    MANAGE: frozenset[Operation] = frozenset(
        {
            Operation.UPDATE_MEMORY,
            Operation.DELETE_MEMORY,
        }
    )

    async def has_permission(
        self, user: AbstractBaseUser | AnonymousUser | None, operation: Operation, **kwargs: Any
    ) -> bool:
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
        if user is None or not user.is_authenticated:
            return operation in self.READ and obj.is_public

        ownership = await obj.memory_users.filter(user=user).afirst()  # ty: ignore[invalid-attribute-access]
        if ownership is None:
            if operation in self.READ | self.WRITE and obj.is_public:
                return True
            return False

        if operation in self.MANAGE:
            return ownership.can_manage
        return True

    def get_queryset_perms(
        self,
        user: UserType,
        operation: Operation,
        queryset: QuerySet,
    ) -> QuerySet:
        is_authenticated = user is not None and getattr(user, "is_authenticated", False)

        if not is_authenticated:
            return queryset.filter(is_public=True) if operation in self.READ else queryset.none()
        if operation in self.READ or operation in self.WRITE:
            return queryset.filter(is_public=True) | queryset.filter(memory_users__user=user)
        if operation in self.MANAGE:
            return queryset.filter(memory_users__user=user)

        return queryset.none()
