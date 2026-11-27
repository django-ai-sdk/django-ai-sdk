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
    from collections.abc import Awaitable, Callable

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.adapters.interfaces import Streamable
    from django_ai_sdk.agents.base import Agent
    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.oversight.plugin import TurnRequest
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
    from django_ai_sdk.oversight.history import history_for
    from django_ai_sdk.oversight.plugin import TurnRequest, mask_stored_question

    sent, vouched = await history_for(
        agent, storage_adapter if thread else None, messages, channel="chat", user=user
    )
    if sent is not messages:
        messages = sent
        newest = bool(messages) and messages[-1].role == "user"
        await agent._resolve_attachments(
            messages[:-1] if newest else messages, thread, user, inline_images=False
        )

    if agent.max_history and len(messages) > agent.max_history:
        messages = messages[-agent.max_history :]
        logger.debug(
            f"Applied max_history={agent.max_history}, kept {len(messages)} most recent messages"
        )

    async def admit(request: TurnRequest) -> None:
        mask_stored_question(messages, request)
        question_id = await _store_question(messages, storage_adapter, thread_id)
        if request.guard is not None and question_id:
            request.guard.user_message_id = question_id
        if agent.title_generation and thread and not thread.title:
            try:
                title = await generate_thread_title(
                    agent=agent, messages=messages, thread_id=thread.id, user=user
                )
                if title:
                    await ThreadService.update_thread(thread.id, title=title, user=user)
            except Exception:
                logger.exception("Thread title generation failed")

    async def build_adapter() -> Streamable:
        adapter = await agent.get_pipeline_adapter(thread_id=thread_id, user=user)
        suggestion_generator = agent.get_suggestion_generator()
        if suggestion_generator:
            adapter.suggestion_generator = suggestion_generator
        context: dict[str, Any] = {}
        if required_tools := agent.get_run_required_tools(messages):
            context["required_tools"] = required_tools
        if thread_id:
            adapter.thread_id = thread_id
            adapter.user = user
            adapter.tool_artifacts = agent.tool_artifacts

            registry = getattr(adapter, "citation_registry", None)
            if registry is not None:
                await registry.seed_from_thread(thread_id)
                registry.seed(await agent.get_citation_sources(thread_id, user))
                adapter.citation_agent = get_citation_agent(agent)
        from django_ai_sdk.adapters.base import Stream

        if isinstance(adapter, Stream) and adapter.agent_component is not None:
            adapter.turn_request = TurnRequest(
                agent=agent,
                channel="chat",
                user=user,
                thread_id=thread_id,
                history_vouched=vouched,
                on_gated=admit,
            )
            adapter.hook_context = context or None
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
    admit: Callable[[TurnRequest], Awaitable[None]],
    user: AbstractBaseUser | AnonymousUser | None,
    thread_id: str | None,
) -> Any:
    from django_ai_sdk.oversight.guard import TurnGuard
    from django_ai_sdk.oversight.plugin import TurnRequest
    from django_ai_sdk.oversight.stream import close_unguardable

    guard = await TurnGuard.start(
        agent, channel="chat", user=user, thread_id=thread_id, messages=messages
    )
    if guard.active and (blocked_stream := await close_unguardable(guard)) is not None:
        return blocked_stream
    await admit(TurnRequest())
    return adapter


async def _store_question(
    messages: list[ChatMessage],
    storage_adapter: BaseStorageAdapter | None,
    thread_id: str | None,
) -> str:
    if not (storage_adapter and thread_id):
        return ""
    question = last_user_message(messages)
    if question is None:
        logger.debug("No user messages found to store")
        return ""
    logger.debug(f"Storing last user message: {len(question.content)} characters")
    question.id = str(uuid.uuid4())
    await storage_adapter.store_chat_message(question)
    return question.id


__all__ = ["chat_turn"]
