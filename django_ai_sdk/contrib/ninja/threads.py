"""Threads, messages, traces and the streaming chat endpoint."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from django.http import HttpRequest
from ninja import Router

from django_ai_sdk.agents.services import AgentService
from django_ai_sdk.contrib.ninja.routing import ApiRouter, Limit, Offset
from django_ai_sdk.contrib.ninja.schemas import (
    CreateThreadResponse,
    DeleteAllThreadsResponse,
    MessageResponse,
    RunResponse,
    Success,
    ThreadDetailResponse,
    ThreadFileMeta,
    ThreadListItem,
    ThreadListResponse,
    ThreadTracesResponse,
)
from django_ai_sdk.errors import NotFound
from django_ai_sdk.memories.services import MemoryService
from django_ai_sdk.permissions import Operation
from django_ai_sdk.storage.services import ThreadService, aget_thread_file_meta, aget_thread_history
from django_ai_sdk.tracing.schemas import TokenUsage
from django_ai_sdk.tracing.services import TraceService
from django_ai_sdk.views.permissions import athread_permissions
from django_ai_sdk.views.schemas import ChatRequest, PatchThreadPayload, RateMessagePayload

routes = ApiRouter()


@routes.get("/threads/", response=ThreadListResponse)
async def list_threads(
    request: HttpRequest,
    limit: Limit = 100,
    offset: Offset = 0,
) -> Any:
    threads = await ThreadService.threads(user=request.user, limit=limit, offset=offset)
    return ThreadListResponse(
        threads=[
            ThreadListItem(
                id=t.id,
                title=t.title,
                agent_id=t.agent_id,
                created_at=t.created_at.isoformat(),
                updated_at=t.updated_at.isoformat(),
                message_count=t.message_count,
            )
            for t in threads
        ]
    )


@routes.post("/threads/", response=CreateThreadResponse)
async def create_thread(request: HttpRequest, payload: ChatRequest) -> Any:
    agent_id = payload.agent_id or ""
    # Initial messages are not persisted here; the chat endpoint receives and stores
    # the full message list.
    thread_id = await ThreadService.create_thread(agent_id=agent_id, user=request.user)
    await MemoryService.link_memories(agent_id, thread_id, user=request.user)
    return CreateThreadResponse(thread_id=thread_id)


@routes.get("/threads/{thread_id}/", response=ThreadDetailResponse)
async def get_thread_history(request: HttpRequest, thread_id: str) -> Any:
    data = await aget_thread_history(thread_id, user=request.user)
    # Each caller sees only their own feedback on a message.
    user_pk = str(request.user.pk) if request.user.is_authenticated else None
    for message in data.get("messages", []):
        feedbacks = message.pop("feedbacks", [])
        message["feedback"] = next((fb for fb in feedbacks if fb.get("user_id") == user_pk), None)
    perms = await athread_permissions(request.user, thread_id)
    return ThreadDetailResponse(**data, permissions=perms)


@routes.post("/threads/{thread_id}/")
async def add_message_to_thread(request: HttpRequest, thread_id: str, payload: ChatRequest) -> Any:
    """The chat endpoint: streams the agent's reply (SSE)."""
    agent = await AgentService.get_agent(thread_id, user=request.user)
    return await agent.as_view(payload.messages, thread_id=thread_id, user=request.user)


@routes.patch("/threads/{thread_id}/", response=Success)
async def patch_thread(request: HttpRequest, thread_id: str, payload: PatchThreadPayload) -> Any:
    """Rename the thread and/or switch it to another agent, moving the linked memories along."""
    if payload.agent_id:
        await AgentService.get(payload.agent_id)
    thread = await ThreadService.get_thread(thread_id, user=request.user)
    if thread is None:
        raise NotFound("Thread not found")
    if payload.agent_id:
        if thread.agent_id:
            await MemoryService.unlink_memories(thread.agent_id, thread_id, user=request.user)
        await ThreadService.update_thread(
            thread_id, metadata={"agent_id": payload.agent_id}, user=request.user
        )
        await MemoryService.link_memories(payload.agent_id, thread_id, user=request.user)
    if payload.title:
        await ThreadService.update_thread(thread_id, title=payload.title, user=request.user)
    return Success(success=True)


@routes.delete("/threads/{thread_id}/", response=Success)
async def delete_thread(request: HttpRequest, thread_id: str) -> Any:
    if not await ThreadService.delete_thread(thread_id, user=request.user):
        raise NotFound("Thread not found")
    return Success(success=True, message="Thread deleted successfully")


@routes.delete("/threads/", response=DeleteAllThreadsResponse)
async def delete_all_threads(request: HttpRequest) -> Any:
    deleted_count = await ThreadService.delete_all_threads(user=request.user)
    return DeleteAllThreadsResponse(success=True, deleted_count=deleted_count)


@routes.post("/threads/{thread_id}/run/", response=RunResponse)
async def run_thread(request: HttpRequest, thread_id: str, payload: ChatRequest) -> Any:
    """Run the thread's agent to completion and return the whole reply at once."""
    agent = await AgentService.get_agent(thread_id, user=request.user)
    await AgentService.has_perms(request.user, Operation.CHAT, agent=agent)
    chat_messages = agent.protocol_handler.to_chat_messages(payload.messages)
    result = await agent.run(chat_messages, thread_id=thread_id, user=request.user)
    return RunResponse(result=result, thread_id=thread_id)


@routes.get("/threads/{thread_id}/file-meta/", response=ThreadFileMeta)
async def get_thread_file_meta(request: HttpRequest, thread_id: str) -> Any:
    return ThreadFileMeta(**await aget_thread_file_meta(thread_id, user=request.user))


@routes.post("/threads/{thread_id}/messages/{message_id}/rate/", response=MessageResponse)
async def rate_message(
    request: HttpRequest, thread_id: str, message_id: str, payload: RateMessagePayload
) -> Any:
    await ThreadService.rate_message(
        thread_id, message_id, payload.rating, feedback=payload.feedback, user=request.user
    )
    return MessageResponse(id=message_id, is_deleted=False)


@routes.post("/threads/{thread_id}/messages/{message_id}/delete/", response=MessageResponse)
async def delete_message(request: HttpRequest, thread_id: str, message_id: str) -> Any:
    await ThreadService.delete_message(thread_id, message_id, user=request.user)
    return MessageResponse(id=message_id, is_deleted=True)


@routes.post("/threads/{thread_id}/messages/{message_id}/restore/", response=MessageResponse)
async def restore_message(request: HttpRequest, thread_id: str, message_id: str) -> Any:
    await ThreadService.restore_message(thread_id, message_id, user=request.user)
    return MessageResponse(id=message_id, is_deleted=False)


@routes.get("/threads/{thread_id}/traces/", response=ThreadTracesResponse)
async def get_thread_traces(
    request: HttpRequest,
    thread_id: str,
    message_id: str | None = None,
    operation_name: str | None = None,
    limit: Limit = 100,
    offset: Offset = 0,
) -> Any:
    traces = await TraceService.thread_traces(
        thread_id,
        user=request.user,
        message_id=message_id,
        operation_name=operation_name,
        limit=limit,
        offset=offset,
    )
    return ThreadTracesResponse(traces=traces)


@routes.get("/threads/{thread_id}/tokens/", response=TokenUsage)
async def get_thread_token_usage(request: HttpRequest, thread_id: str) -> Any:
    return await TraceService.thread_token_usage(thread_id, user=request.user)


@routes.get("/messages/{message_id}/traces/", response=ThreadTracesResponse)
async def get_message_traces(
    request: HttpRequest,
    message_id: str,
    operation_name: str | None = None,
    limit: Limit = 100,
    offset: Offset = 0,
) -> Any:
    traces = await TraceService.message_traces(
        message_id, user=request.user, operation_name=operation_name, limit=limit, offset=offset
    )
    return ThreadTracesResponse(traces=traces)


@routes.get("/messages/{message_id}/tokens/", response=TokenUsage)
async def get_message_token_usage(request: HttpRequest, message_id: str) -> Any:
    return await TraceService.message_token_usage(message_id, user=request.user)


def get_threads_router(*, exclude: Collection[str] = (), **router_kwargs: Any) -> Router:
    """A new Router with the thread endpoints; mount with ``api.add_router("/", ...)``."""
    return routes.build(exclude=exclude, **router_kwargs)
