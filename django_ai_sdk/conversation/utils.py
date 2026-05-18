from typing import TYPE_CHECKING

from django_ai_sdk.common import ChatMessage
from django_ai_sdk.prompts import TITLE_GENERATION_PROMPT

if TYPE_CHECKING:
    from django_ai_sdk.assistant import Assistant


async def generate_thread_title(assistant: "Assistant", messages: list[ChatMessage]) -> str | None:
    """Extract a thread title from the user message(s)."""
    return await assistant.run(
        messages=messages,
        system_prompt=TITLE_GENERATION_PROMPT,
    )
