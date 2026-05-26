from typing import TYPE_CHECKING, TypedDict

from django_ai_sdk.assistants.registry import registry

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

    All methods are static. No instance state is required.
    """

    @staticmethod
    def from_registry(assistant_id: str) -> "Assistant":
        """Resolve assistant from registry or raise ValueError."""
        assistant = registry.get(assistant_id)
        if assistant is None:
            raise ValueError(f"Assistant '{assistant_id}' not found")
        return assistant

    @staticmethod
    async def get_assistant(thread_id: str, user: AbstractUser | None = None) -> "Assistant":
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
    def list_assistants() -> list[AssistantSummary]:
        """Return all registered assistants as a list of typed dicts."""
        from django_ai_sdk.assistants.registry import registry

        return [
            AssistantSummary(
                id=aid,
                name=a.name,
                model=a.model,
            )
            for aid, a in registry.all().items()
        ]
