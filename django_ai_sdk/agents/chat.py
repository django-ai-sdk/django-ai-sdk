"""One chat turn of an agent: pick the history, stream the run, store the question once it passed."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from django_ai_sdk.adapters.citations.grounding import get_citation_agent
from django_ai_sdk.common import last_user_message
from django_ai_sdk.conversation.utils import generate_thread_title
from django_ai_sdk.logger import get_logger
from django_ai_sdk.responses import stream_response
from django_ai_sdk.storage.services import ThreadService

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.adapters.interfaces import Streamable
    from django_ai_sdk.agents.base import Agent
    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.oversight.plugin import Turn
    from django_ai_sdk.storage.base import BaseStorageAdapter
    from django_ai_sdk.storage.schemas import ThreadInfo

logger = get_logger(__name__)


async def chat_turn(
    agent: Agent,
    messages: list[ChatMessage],
    thread: ThreadInfo | None,
    thread_id: str | None,
    storage_adapter: BaseStorageAdapter | None,
    user: AbstractBaseUser | AnonymousUser | None,
) -> Any:
    """Pick the history, then stream. The run's hooks open the turn and gate it."""
    from django_ai_sdk.oversight.history import history_for
    from django_ai_sdk.oversight.plugin import keep_masked

    # Prefer the stored thread over the client's history. The client's copy may
    # be unmasked or made up. If there is no usable stored copy, the client's
    # history is judged as sent.
    sent, vouched = await history_for(
        agent, storage_adapter if thread else None, messages, channel="chat", user=user
    )
    if sent is not messages:
        messages = sent
        newest = bool(messages) and messages[-1].role == "user"
        # Validate the stored messages' attachments too.
        await agent._resolve_attachments(
            messages[:-1] if newest else messages, thread, user, inline_images=False
        )

    if agent.max_history and len(messages) > agent.max_history:
        messages = messages[-agent.max_history :]
        logger.debug(
            f"Applied max_history={agent.max_history}, kept {len(messages)} most recent messages"
        )

    async def admit(turn: Turn | None) -> None:
        """Store the question as the gate left it, then title the thread.

        Oversight: the run's gate hook calls this once the question passed, so a
        refused question is never stored.
        """
        keep_masked(messages, turn)
        question_id = await _store_question(messages, storage_adapter, thread_id)
        if turn is not None and question_id:
            turn.guard.user_message_id = question_id
        if agent.title_generation and thread and not thread.title:
            # A failed title must not fail the reply.
            # Oversight: off on purpose. The title uses the question after the gate.
            try:
                title = await generate_thread_title(
                    agent=agent, messages=messages, thread_id=thread.id, user=user
                )
                if title:
                    await ThreadService.update_thread(thread.id, title=title, user=user)
            except Exception:
                logger.exception("Thread title generation failed")

    async def build_adapter() -> Streamable:
        # Built inside the stream, so a failure (e.g. RAG warmup) reaches the client.
        adapter = await agent.get_pipeline_adapter(thread_id=thread_id, user=user)
        suggestion_generator = agent.get_suggestion_generator()
        if suggestion_generator:
            adapter.suggestion_generator = suggestion_generator
        context: dict[str, Any] = {}
        if required_tools := agent.get_run_required_tools(messages):
            context["required_tools"] = required_tools
        # Artifacts are stored on the thread: none without one.
        if thread_id:
            adapter.thread_id = thread_id
            adapter.user = user
            # Shown after each mapped tool's result, for any tool of the run.
            adapter.tool_artifacts = agent.tool_artifacts

            # Seed the citation registry with sources from current thread
            registry = getattr(adapter, "citation_registry", None)
            if registry is not None:
                await registry.seed_from_thread(thread_id)
                registry.seed(await agent.get_citation_sources(thread_id, user))
                adapter.citation_agent = get_citation_agent(agent)
        from django_ai_sdk.adapters.base import Stream

        if isinstance(adapter, Stream) and adapter.agent_component is not None:
            # Oversight: the run's hooks open, gate, check and finish the turn
            # (see oversight.plugin); the Stream shows what they decided.
            adapter.hook_context = {
                **context,
                "agent": agent,
                "channel": "chat",
                "user": user,
                "thread_id": thread_id,
                "history_vouched": vouched,
                "on_gated": admit,
            }
            return adapter
        adapter.hook_context = context or None
        return await _unguardable(agent, adapter, messages, admit, user, thread_id)

    logger.debug("Initiating stream response")
    return await stream_response(
        build_adapter,
        messages,
        agent.protocol_handler,
        storage_adapter=storage_adapter,
        warmup=agent.warmup_progress(thread_id, user),
    )


async def _unguardable(
    agent: Agent,
    adapter: Any,
    messages: list[ChatMessage],
    admit: Any,
    user: AbstractBaseUser | AnonymousUser | None,
    thread_id: str | None,
) -> Any:
    """An adapter that is no Haystack Agent: refused under a policy, unless it fails open.

    Oversight: nothing can check its tool calls or its answer.
    """
    from django_ai_sdk.oversight.guard import TurnGuard
    from django_ai_sdk.oversight.stream import close_unguardable

    guard = await TurnGuard.start(
        agent, channel="chat", user=user, thread_id=thread_id, messages=messages
    )
    if guard.active and (refusal := await close_unguardable(guard)) is not None:
        return refusal
    await admit(None)
    return adapter


async def _store_question(
    messages: list[ChatMessage],
    storage_adapter: BaseStorageAdapter | None,
    thread_id: str | None,
) -> str:
    """Store only the last user message, the one just sent, and return its id.

    Many protocols send the full conversation, but only the new message is stored.
    """
    if not (storage_adapter and thread_id):
        return ""
    question = last_user_message(messages)
    if question is None:
        logger.debug("No user messages found to store")
        return ""
    logger.debug(f"Storing last user message: {len(question.content)} characters")
    # FIXME: we need a better way on how to create a id for the user messages
    question.id = str(uuid.uuid4())
    await storage_adapter.store_chat_message(question)
    return question.id


__all__ = ["chat_turn"]
