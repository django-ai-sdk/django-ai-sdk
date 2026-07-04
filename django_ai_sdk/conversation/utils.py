from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.assistant import Assistant
    from django_ai_sdk.common import ChatMessage


async def generate_thread_title(
    assistant: Assistant,
    messages: list[ChatMessage],
    thread_id: str | None = None,
    user: AbstractBaseUser | AnonymousUser | None = None,
) -> str | None:
    """Extract a thread title from the user message(s)."""
    return await assistant.run(
        messages=messages,
        system_prompt=assistant.get_title_generation_prompt(),
        thread_id=thread_id,
        user=user,
    )
