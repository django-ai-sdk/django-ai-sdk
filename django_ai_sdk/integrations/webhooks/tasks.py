"""Worker task that runs the agent and posts the answer back."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any

from asgiref.sync import async_to_sync
from django_tasks import task

from django_ai_sdk.common import ChatMessage
from django_ai_sdk.integrations.registry import get_integrations
from django_ai_sdk.integrations.webhooks.base import (
    InboundEvent,
    WebhookIntegration,
    current_event,
)
from django_ai_sdk.memories.services import MemoryService
from django_ai_sdk.storage.services import thread_metadata
from django_ai_sdk.utils import resolve_setting

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser

    from django_ai_sdk.agents.base import Agent
    from django_ai_sdk.storage.base import BaseStorageAdapter

logger = logging.getLogger(__name__)

TIMEOUT_MESSAGE = "Sorry, that took too long. Try asking something smaller."
ERROR_MESSAGE = "Sorry, something went wrong answering that."

# Fixed namespace, so a conversation always maps to the same thread id.
_THREAD_NAMESPACE = uuid.UUID("6b1f3c2e-9a84-4d1f-b7e0-2c8a5d9f1e44")


@task(queue_name="default")
def handle_inbound(payload: dict[str, Any]) -> None:
    """Sync entry point for the worker."""
    async_to_sync(run_inbound)(payload)


async def run_inbound(payload: dict[str, Any]) -> None:
    """Answer one inbound event. Always replies once an agent is found."""
    event = InboundEvent.model_validate(payload)
    integrations = await get_integrations([event.integration])
    integration = integrations.get(event.integration)
    # Check again here. The integration may have been removed since the request.
    if not isinstance(integration, WebhookIntegration):
        logger.warning(
            "Dropping event for %r, which no longer receives webhooks", event.integration
        )
        return

    agent = integration.resolve_agent()
    if agent is None:
        # Still reply, so the person in the channel is not left waiting.
        logger.error(
            "AI_SDK_INTEGRATIONS[%r]['AGENT'] does not resolve to a registered agent; "
            "answering with the error message",
            event.integration,
        )
        await integration.reply(event, ERROR_MESSAGE)
        return

    # SECURITY: never trust the platform's sender id as a user. Runs act as the
    # RUN_AS account, and tools, threads and memories are checked against it.
    # Without RUN_AS the run is anonymous and reaches no integration.
    principal = await integration.aget_principal()

    storage, thread_id, history = await _load_conversation(agent, integration, event, principal)
    user_message = ChatMessage(role="user", content=event.text)
    messages = [*history, user_message]

    timeout = resolve_setting("AI_SDK_WEBHOOK_TIMEOUT", 120)
    token = current_event.set(event)
    try:
        text = await asyncio.wait_for(
            agent.run(messages, tools=True, user=principal, thread_id=thread_id),
            timeout=timeout,
        )
    except TimeoutError:
        logger.warning(
            "Agent %r timed out after %s seconds on a %r event",
            agent.name,
            timeout,
            event.integration,
        )
        text = TIMEOUT_MESSAGE
    except Exception:
        # Log it, since the generic reply tells the deployer nothing.
        logger.exception("Agent run for a %r event failed", event.integration)
        text = ERROR_MESSAGE
    finally:
        current_event.reset(token)

    reply_text = str(text) if text else ERROR_MESSAGE
    # Store fallback replies too, so the thread matches what the channel shows.
    if storage is not None:
        await _persist_turn(storage, user_message, reply_text)
    await integration.reply(event, reply_text)


async def _load_conversation(
    agent: Agent,
    integration: WebhookIntegration,
    event: InboundEvent,
    principal: AbstractBaseUser | None,
) -> tuple[BaseStorageAdapter | None, str | None, list[ChatMessage]]:
    """Create the thread if needed and return its earlier messages."""
    scope = integration.thread_scope(event)
    storage_class = agent.storage_adapter
    if not scope or storage_class is None:
        return None, None, []

    thread_id = str(uuid.uuid5(_THREAD_NAMESPACE, scope))
    if await storage_class.get_thread(thread_id) is None:
        try:
            await storage_class.create_thread(
                title=f"{integration.label}: {event.conversation_id}",
                metadata={
                    **thread_metadata(agent.__class__._agent_id, agent, "webhook"),
                    "integration": event.integration,
                    "conversation_id": event.conversation_id,
                    "thread_ref": event.thread_ref,
                    "external_user_id": event.external_user_id,
                    "webhook_scope": scope,
                },
                # Owned by the run's user, so normal thread permissions apply.
                user=principal,
                thread_id=thread_id,
            )
        except Exception:
            # Two events can race to create the same thread. If the thread now
            # exists, use it. Storage may not be a database, so any error is
            # checked this way rather than only integrity errors.
            if await storage_class.get_thread(thread_id) is None:
                raise
        else:
            # Link the agent's memories to the new thread, as the thread endpoints do.
            await MemoryService.link_memories(agent.__class__._agent_id, thread_id, user=principal)

    storage = storage_class(thread_id)
    history = await storage.get_messages()
    # Agent.run sends what it is given, so apply max_history here.
    if agent.max_history and len(history) > agent.max_history:
        history = history[-agent.max_history :]
    return storage, thread_id, history


async def _persist_turn(
    storage: BaseStorageAdapter,
    user_message: ChatMessage,
    reply_text: str,
) -> None:
    """Store the user message and the reply on the thread."""
    # No transaction, since storage may not be a database. A half-written turn
    # looks unanswered next time, so log the failure loudly.
    try:
        if not user_message.id:
            user_message.id = str(uuid.uuid4())
        await storage.store_chat_message(user_message)
        await storage.store_chat_message(
            ChatMessage(role="assistant", content=reply_text, id=str(uuid.uuid4()))
        )
    except Exception:
        logger.exception("Failed to persist webhook turn on thread %s", storage.thread_id)


__all__ = ["ERROR_MESSAGE", "TIMEOUT_MESSAGE", "handle_inbound", "run_inbound"]
