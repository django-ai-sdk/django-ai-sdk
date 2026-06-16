from __future__ import annotations

from typing import TYPE_CHECKING

from django.conf import settings

from django_ai_sdk.common import get_logger

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.assistant import Assistant
    from django_ai_sdk.common import ChatMessage

logger = get_logger(__name__)

# THREAD_TITLE_MAX_LENGTH (255) is the DB column's storage limit, not a
# meaningful signal that generated text "looks like a title" — a model that
# ignores the prompt and echoes back a whole response can easily stay under
# 255 chars while still being nothing like a title. Real titles are short
# (a handful of words); this catches that class of bad output specifically.
TITLE_SANITY_LIMIT_DEFAULT = 80


def get_title_sanity_limit() -> int:
    """Overridable via `AI_SDK_TITLE_SANITY_LIMIT` setting."""
    return getattr(settings, "AI_SDK_TITLE_SANITY_LIMIT", TITLE_SANITY_LIMIT_DEFAULT)


_MAX_TITLE_LENGTH = 60


async def generate_thread_title(
    assistant: Assistant,
    messages: list[ChatMessage],
    thread_id: str | None = None,
    user: AbstractBaseUser | AnonymousUser | None = None,
) -> str | None:
    """Extract a thread title from the user message(s).

    Returns None if generation fails or produces something that isn't
    title-shaped (e.g. the model ignores the prompt and echoes back the full
    conversation instead of a short title). Callers should treat None as "no
    title yet" rather than storing it, so generation is retried on the next
    message instead of a bad title getting stuck permanently.

    For now resolving only user messages. The assistant message can hold too
    much context if e.g. a MCP fills the full assistant context.
    """
    user_messages = [m for m in messages if m.role == "user"]
    if not user_messages:
        return None

    try:
        title = await assistant.run(
            messages=user_messages,
            system_prompt=assistant.get_title_generation_prompt(),
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
