from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from asgiref.sync import async_to_sync

from django_ai_sdk.assistants.registry import registry
from django_ai_sdk.permissions import AllowAll, Operation, PermissionDenied, check_permissions

if TYPE_CHECKING:
    from django.contrib.auth.models import AbstractUser

    from django_ai_sdk.assistant import Assistant


class AssistantSummary(TypedDict):
    id: str
    name: str | None
    model: str | None


class AssistantService:
    """
    Service for resolving assistants from the registry.
    """

    @staticmethod
    def from_registry(assistant_id: str) -> Assistant:
        """Resolve assistant from registry or raise ValueError."""
        assistant = registry.get(assistant_id)
        if assistant is None:
            raise ValueError(f"Assistant '{assistant_id}' not found")
        return assistant

    @staticmethod
    async def get_assistant(thread_id: str, user: AbstractUser | None = None) -> Assistant:
        """Find the thread and return its associated assistant.

        Args:
            thread_id: Thread ID to look up
            user: Optional user for permission checking on the thread lookup

        Returns:
            The assistant instance associated with the thread

        Raises:
            ValueError: If thread or assistant not found
        """
        from django_ai_sdk.storage.services import ThreadService

        thread = await ThreadService.get_thread(thread_id, user=user)
        if thread is None:
            raise ValueError("Thread not found")
        return AssistantService.from_registry(thread.assistant_id)

    @staticmethod
    async def list_assistants(user: AbstractUser | None = None) -> list[AssistantSummary]:
        """Return all registered assistants the user is allowed to view."""
        result: list[AssistantSummary] = []
        for aid, assistant in registry.all().items():
            perm_classes: list = getattr(assistant, "permissions", [AllowAll])
            try:
                await check_permissions(user, Operation.VIEW_ASSISTANT, perm_classes)
                result.append(
                    AssistantSummary(id=aid, name=assistant.name, model=assistant.model)
                )
            except PermissionDenied:
                continue
        return result

    @staticmethod
    async def get_assistant_info(
        assistant_id: str, user: AbstractUser | None = None
    ) -> dict:
        """Return assistant info if user has VIEW_ASSISTANT permission."""
        assistant = AssistantService.from_registry(assistant_id)
        perm_classes: list = getattr(assistant, "permissions", [AllowAll])
        await check_permissions(user, Operation.VIEW_ASSISTANT, perm_classes)
        return assistant.info()


list_assistants = async_to_sync(AssistantService.list_assistants)
get_assistant_info = async_to_sync(AssistantService.get_assistant_info)
