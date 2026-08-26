"""Where a workflow's output goes once every step has run.

A host declares its own implementations by key in AI_SDK_WORKFLOW_ACTIONS.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from django.utils.module_loading import import_string

from django_ai_sdk.utils import resolve_setting

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

logger = logging.getLogger(__name__)


@dataclass
class ActionContext:
    """Who and what an action is delivering for.

    Every field is optional: an owned run and an app-level one share one signature.
    """

    user: AbstractBaseUser | AnonymousUser | None = None
    # Produced the payload, so its storage adapter opens the thread.
    agent_id: str = ""
    # Human-readable origin, used for titles and logs.
    source: str = ""


@runtime_checkable
class BaseAction(Protocol):
    description: str

    async def execute(self, payload: Any, context: ActionContext) -> None: ...


class ThreadMessageAction:
    """Deliver the payload into a new chat thread owned by `context.user`.

    Nobody to deliver to logs and returns: the steps already ran.
    """

    description = "Post the result into a new chat thread for the run's user"

    async def execute(self, payload: Any, context: ActionContext) -> None:
        from django_ai_sdk.common import ChatMessage
        from django_ai_sdk.storage.services import ThreadService

        if context.user is None or context.user.is_anonymous:
            logger.warning(
                "thread_message: %s has no user to own a thread; skipping.", context.source
            )
            return
        if not context.agent_id:
            logger.warning(
                "thread_message: %s has no agent to deliver as; skipping.", context.source
            )
            return

        content = _as_text(payload)
        thread_id = await ThreadService.create_thread(
            context.agent_id,
            title=context.source,
            metadata={"created_via": "action:thread_message"},
            user=context.user,
        )
        storage = await ThreadService.storage_for_thread(thread_id, user=context.user)
        await storage.store_chat_message(
            # Adapters take an id rather than minting one.
            ChatMessage(id=str(uuid.uuid4()), role="assistant", content=content)
        )
        await _retitle(thread_id, content, context)


async def _retitle(thread_id: str, content: str, context: ActionContext) -> None:
    """Replace the fallback title with a generated one.

    Returns quietly when the agent is gone or opts out of titles.
    """
    from django_ai_sdk.agents.services import AgentService
    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.conversation.utils import generate_thread_title
    from django_ai_sdk.storage.services import ThreadService

    try:
        agent = await AgentService.get(context.agent_id)
    except ValueError:
        return
    if not agent.title_generation:
        return

    title = await generate_thread_title(
        agent=agent,
        messages=[ChatMessage(id=str(uuid.uuid4()), role="user", content=content)],
        thread_id=thread_id,
        user=context.user,
    )
    if title:
        await ThreadService.update_thread(thread_id, title=title, user=context.user)


def _as_text(payload: Any) -> str:
    """Render an action payload as message text, JSON rather than a Python repr."""
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, default=str, indent=2)


@lru_cache(maxsize=1)
def get_action_registry() -> dict[str, type[BaseAction]]:
    """Built-in actions, overlaid with those declared in AI_SDK_WORKFLOW_ACTIONS."""
    registry: dict[str, type[BaseAction]] = {"thread_message": ThreadMessageAction}
    extra: dict[str, str] = resolve_setting("AI_SDK_WORKFLOW_ACTIONS", {})
    for key, path in extra.items():
        registry[key] = import_string(path)
    return registry
