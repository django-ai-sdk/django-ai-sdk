from __future__ import annotations

from typing import TYPE_CHECKING

from django_ai_sdk.common import get_logger
from django_ai_sdk.utils import resolve_setting

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.agent import Agent
    from django_ai_sdk.common import ChatMessage

logger = get_logger(__name__)

# Default title length for generated thread titles.
TITLE_SANITY_LIMIT_DEFAULT = 80


def get_title_sanity_limit() -> int:
    """Overridable via `AI_SDK_TITLE_SANITY_LIMIT` setting."""
    return resolve_setting("AI_SDK_TITLE_SANITY_LIMIT", TITLE_SANITY_LIMIT_DEFAULT)


async def generate_thread_title(
    agent: Agent,
    messages: list[ChatMessage],
    thread_id: str | None = None,
    user: AbstractBaseUser | AnonymousUser | None = None,
) -> str | None:
    """Extract a thread title from the user message(s)."""

    user_messages = [m for m in messages if m.role == "user"]
    if not user_messages:
        return None

    try:
        title = await agent.run(
            messages=user_messages,
            system_prompt=agent.get_title_generation_prompt(),
            thread_id=thread_id,
            user=user,
            response_format=None,
        )
    except Exception:
        logger.warning("Thread title generation failed", exc_info=True)
        return None

    title = (title or "").strip()
    if not title or len(title) > get_title_sanity_limit():
        logger.warning(
            "Thread title generation produced invalid output (length=%d): %r",
            len(title),
            title[:80],
        )
        return None

    return title
