"""Background half of the webhook loop: run the agent, post the answer back."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any

from asgiref.sync import async_to_sync
from django_tasks import task

from django_ai_sdk.common import ChatMessage
from django_ai_sdk.integrations.registry import get_integrations
from django_ai_sdk.integrations.webhooks.base import InboundEvent, WebhookIntegration
from django_ai_sdk.utils import resolve_setting

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser

    from django_ai_sdk.agent import Agent
    from django_ai_sdk.storage.base import BaseStorageAdapter

logger = logging.getLogger(__name__)

TIMEOUT_MESSAGE = "Sorry, that took too long. Try asking something smaller."
ERROR_MESSAGE = "Sorry, something went wrong answering that."

# Fixed namespace so the same platform conversation resolves to the same thread id
# across processes and restarts, without a lookup table.
_THREAD_NAMESPACE = uuid.UUID("6b1f3c2e-9a84-4d1f-b7e0-2c8a5d9f1e44")


@task(queue_name="default")
def handle_inbound(payload: dict[str, Any]) -> None:
    """Sync task entry point — the worker calls this, it bridges to async."""
    async_to_sync(run_inbound)(payload)


async def run_inbound(payload: dict[str, Any]) -> None:
    """Answer one inbound event, replying on every path that reaches the agent."""
    event = InboundEvent.model_validate(payload)
    integrations = await get_integrations([event.integration])
    integration = integrations.get(event.integration)
    # The same isinstance guard the view applies, because dispatch is at-least-once
    # and an integration can be uninstalled between the request and the worker.
    if not isinstance(integration, WebhookIntegration):
        logger.warning(
            "Dropping event for %r, which no longer receives webhooks", event.integration
        )
        return

    agent = integration.resolve_agent()
    if agent is None:
        return

    # SECURITY: the platform's sender id is not a principal. RUN_AS names the account
    # this integration's runs act as, and everything the run touches — tools, threads,
    # memories — is checked against it. Unset answers anonymously, which reaches no
    # integration, rather than silently running with the deployer's reach.
    principal = await integration.aget_principal()

    storage, thread_id, history = await _load_conversation(agent, integration, event, principal)
    user_message = ChatMessage(role="user", content=event.text)
    messages = [*history, user_message]

    timeout = resolve_setting("AI_SDK_WEBHOOK_TIMEOUT", 120)
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
        # Logged rather than swallowed: the reply below is all the person in the
        # channel sees, and it carries nothing a deployer can debug from.
        logger.exception("Agent run for a %r event failed", event.integration)
        text = ERROR_MESSAGE

    reply_text = str(text) if text else ERROR_MESSAGE
    # The fallbacks are persisted too: the channel shows them, so a thread that
    # omitted them would disagree with what the person can scroll back to.
    if storage is not None:
        await _persist_turn(storage, user_message, reply_text)
    await integration.reply(event, reply_text)


async def _load_conversation(
    agent: Agent,
    integration: WebhookIntegration,
    event: InboundEvent,
    principal: AbstractBaseUser | None,
) -> tuple[BaseStorageAdapter | None, str | None, list[ChatMessage]]:
    """Ensure a persisted thread for this webhook scope and return prior messages."""
    scope = integration.thread_scope(event)
    storage_class = agent.storage_adapter
    if not scope or storage_class is None:
        return None, None, []

    thread_id = str(uuid.uuid5(_THREAD_NAMESPACE, scope))
    if await storage_class.get_thread(thread_id) is None:
        await storage_class.create_thread(
            title=f"{integration.label}: {event.conversation_id}",
            metadata={
                "agent_id": agent.__class__._agent_id,
                "model": agent.model,
                "agent_name": agent.name or agent.__class__.__name__,
                "agent_class": agent.__class__.__name__,
                "created_via": "webhook",
                "integration": event.integration,
                "conversation_id": event.conversation_id,
                "thread_ref": event.thread_ref,
                "external_user_id": event.external_user_id,
                "webhook_scope": scope,
            },
            # Owned by the same principal the run acts as, so the thread is reachable
            # under the ordinary thread permissions rather than orphaned.
            user=principal,
            thread_id=thread_id,
        )

    storage = storage_class(thread_id)
    history = await storage.get_messages()
    if agent.max_history and len(history) > agent.max_history:
        history = history[-agent.max_history :]
    return storage, thread_id, history


async def _persist_turn(
    storage: BaseStorageAdapter,
    user_message: ChatMessage,
    reply_text: str,
) -> None:
    """Store the user turn and the assistant reply on the webhook thread."""
    # Two writes, no transaction: a storage adapter is not necessarily a database, so
    # no atomic() spans both. A half-written turn reads as unanswered on the next
    # run, hence the loud log rather than a silent pass.
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
