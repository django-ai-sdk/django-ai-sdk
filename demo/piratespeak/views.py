"""
Pirate Speak API endpoints for the demo application.

This module defines the Ninja API router and all endpoints for the piratespeak demo,
including thread management and message operations.
"""

import uuid
from typing import Any

from django.http import HttpRequest, StreamingHttpResponse
from django_ai_sdk import Assistant
from django_ai_sdk.assistants import AssistantInfo
from django_ai_sdk.assistants.registry import registry
from django_ai_sdk.common import ChatMessage
from django_ai_sdk.protocols.openai import OpenAIProtocolHandler
from django_ai_sdk.storage import ThreadService
from django_ai_sdk.storage.schemas import ThreadInfo
from ninja import Field, Router, Schema
from pydantic import BaseModel

# Create router
router = Router()


# ============================================================================
# Schema Definitions
# ============================================================================


class Error(Schema):
    message: str
    code: int | None = None


class Success(Schema):
    success: bool
    message: str | None = None


class MessagePart(Schema):
    type: str
    text: str | None = None


class Message(Schema):
    role: str
    parts: list[MessagePart]
    id: str | None = None


class ChatRequest(Schema):
    messages: list[Message]
    id: str | None = None
    trigger: str | None = None
    assistant_id: str | None = None


class RateMessagePayload(BaseModel):
    rating: int = Field(1 | -1, description="Rating value: 1 for good, -1 for bad")


class MessageResponse(Schema):
    id: str
    rating: int | None = None
    is_deleted: bool = False


class ThreadListItem(Schema):
    id: str
    title: str
    assistant_id: str
    created_at: str
    updated_at: str
    message_count: int


class ThreadListResponse(Schema):
    threads: list[ThreadListItem]


class CreateThreadResponse(Schema):
    thread_id: str | None = None


class ThreadSiloInfo(Schema):
    """Silo information included in thread history."""

    id: str
    name: str
    description: str
    document_count: int
    active: bool


class ThreadDetailWithSilos(Schema):
    """Thread detail with silos included."""

    thread: ThreadInfo
    silos: list[ThreadSiloInfo]
    messages: list


# ============================================================================
# Health & Info Endpoints
# ============================================================================


@router.get("/health/")
def health_check(request: HttpRequest) -> dict:
    """Simple health check endpoint."""
    return {"status": "ok", "service": "piratespeak"}


@router.get("/assistants/")
def list_assistants(request: HttpRequest) -> dict:
    """List available assistants."""
    return {
        "assistants": [
            {
                "id": assistant_id,
                "name": assistant.name,
                "model": assistant.model,
            }
            for assistant_id, assistant in registry.all().items()
        ]
    }


@router.get("/assistants/{assistant_id}/", response={200: AssistantInfo, 404: Error})
def get_assistant_info(request: HttpRequest, assistant_id: str) -> AssistantInfo | Error:
    """Get detailed information about a specific assistant.

    Args:
        assistant_id: The full UUID of the assistant

    Returns:
        AssistantInfo model with assistant details
    """
    assistant = registry.get(assistant_id)
    if assistant is None:
        return Error(message=f"Assistant '{assistant_id}' not found")

    return assistant.info()


@router.post(
    "/assistants/{assistant_id}/reindex/",
    response={200: Success, 404: Error},
)
async def reindex_assistant(
    request: HttpRequest,
    assistant_id: str,
    silo_id: str | None = None,
    force_rebuild: bool = False,
) -> Success | Error:
    """Reindex the RAG pipeline for an assistant.

    Args:
        assistant_id: The ID of the assistant to reindex
        silo_id: Optional silo ID to limit reindexing to specific documents
        force_rebuild: If True, forces a complete rebuild of the index
                      (clears persistent storage for backends like Qdrant)

    Returns:
        Success status and message
    """
    import logging

    logger = logging.getLogger(__name__)
    logger.info(
        f"[reindex_assistant] assistant_id={assistant_id}, silo_id={silo_id}, force_rebuild={force_rebuild}"
    )

    if assistant_id not in registry:
        return Error(message=f"Assistant '{assistant_id}' not found")

    assistant = registry.get(assistant_id)

    if not assistant:
        return Error(message=f"Assistant '{assistant_id}' not found")

    try:
        result = await Assistant.reindex(assistant, silo_id, force_rebuild)

        if result is None:
            return Success(success=False, message="No RAG provider configured for this assistant")

        rebuild_msg = " (force rebuild)" if force_rebuild else ""
        return Success(
            success=True,
            message="RAG pipeline reindexed successfully"
            + rebuild_msg
            + (f" for silo {silo_id}" if silo_id else ""),
        )
    except Exception as e:
        return Success(success=False, message=f"Reindex failed: {str(e)}")


# ============================================================================
# Thread Management Endpoints
# ============================================================================


@router.get("/threads/", response={200: ThreadListResponse, 404: Error})
async def list_threads(request: HttpRequest) -> ThreadListResponse | Error:
    """
    Get list of all threads with id and title.

    Returns:
        List of threads with basic information for sidebar display.
    """
    threads = await ThreadService.threads(
        user_id=str(request.user.id) if request.user and request.user.id else None  # type: ignore[attr-defined] # Django user id
    )

    items = []
    for thread in threads:
        items.append(
            {
                "id": thread.id,
                "title": thread.title,
                "assistant_id": thread.assistant_id,
                "created_at": thread.created_at.isoformat(),
                "updated_at": thread.updated_at.isoformat(),
                "message_count": thread.message_count,
            }
        )

    return ThreadListResponse(threads=items)


@router.post("/threads/", response={200: CreateThreadResponse, 404: Error})
async def create_thread(request: HttpRequest, payload: ChatRequest) -> CreateThreadResponse | Error:
    """
    Create a new conversation thread.
    This endpoint only creates the thread and returns the thread_id
    """
    assistant_id = getattr(payload, "assistant_id", "pirate_basic")
    assistant = registry.get(assistant_id)

    if not assistant:
        return Error(message=f"Assistant '{assistant_id}' not found")

    # Generate thread title from first user message if provided
    title = "New Conversation"
    if payload.messages:
        for message in payload.messages:
            if message.role == "user" and message.parts:
                for part in message.parts:
                    if part.type == "text" and part.text:
                        # Use first 50 chars of user message as title
                        title = part.text[:50].strip()
                        if len(part.text) > 50:
                            title += "..."
                        break
                break

    # Create thread using ThreadService
    thread_id = await ThreadService.create_thread(
        assistant_id=assistant_id,
        title=title,
        metadata={
            "model": getattr(assistant, "model", "unknown"),
            "assistant_name": getattr(assistant, "name", assistant.__class__.__name__),
            "assistant_class": assistant.__class__.__name__,
            "created_via": "create_thread",
        },
        user_id=str(request.user.id) if request.user and request.user.id else None,  # type: ignore[attr-defined] # Django user id
    )

    return CreateThreadResponse(thread_id=thread_id)


@router.get("/threads/{thread_id}/", response={200: ThreadDetailWithSilos, 404: Error})
async def get_thread_history(request: HttpRequest, thread_id: str) -> ThreadDetailWithSilos | Error:
    """
    Get conversation history for a specific thread.

    Returns thread metadata, connected silos, and message history in JSON format.
    Uses Assistant.history() to support any storage adapter.
    """
    from django.db.models import Count
    from django_ai_sdk.silos.models import ThreadSilo

    # Find thread and assistant using ThreadService
    thread = await ThreadService.get_assistant(thread_id)
    if not thread:
        return Error(message="Thread not found")

    assistant_id = thread.assistant_id
    if assistant_id not in registry:
        assistant_id = "pirate_basic"

    assistant = registry.get(assistant_id)
    if not assistant:
        return Error(message=f"Assistant '{assistant_id}' not found")

    # Get history via assistant (returns ThreadDetail with thread metadata and messages)
    thread_detail = await assistant.history(thread_id)

    # Get connected silos with document count - use async for iteration
    thread_silos_query = (
        ThreadSilo.objects.filter(thread_id=thread_id)
        .select_related("silo")
        .annotate(document_count=Count("silo__documents"))
    )

    silos = []
    async for ts in thread_silos_query:
        silos.append(
            ThreadSiloInfo(
                id=str(ts.silo.id),
                name=ts.silo.name,
                description=ts.silo.description,
                document_count=ts.document_count,
                active=ts.active,
            )
        )

    return ThreadDetailWithSilos(
        thread=thread_detail.thread,
        silos=silos,
        messages=thread_detail.messages,
    )


@router.post("/threads/{thread_id}/", response={404: Error})
async def add_message_to_thread(
    request: HttpRequest, thread_id: str, payload: ChatRequest
) -> StreamingHttpResponse | Error:
    """
    Add a message to an existing thread and get streaming response.

    Args:
        thread_id: The thread ID to add the message to
        payload: Chat request with new messages

    Returns:
        Streaming chat response (SSE)
    """
    # Find thread using ThreadService
    thread = await ThreadService.get_assistant(thread_id)

    if not thread:
        return Error(message="Thread not found")

    # Get the assistant for this thread
    assistant_id = thread.assistant_id
    if assistant_id not in registry:
        assistant_id = "pirate_basic"

    assistant = registry.get(assistant_id)

    if not assistant:
        return Error(message=f"Assistant '{assistant_id}' not found")

    # Use assistant with thread_id for automatic storage
    return await assistant.as_view(payload.messages, thread_id=thread_id)


@router.delete("/threads/{thread_id}/", response={200: Success, 404: Error})
async def delete_thread(request: HttpRequest, thread_id: str) -> Success | Error:
    """
    Delete a conversation thread and all its messages.

    Args:
        thread_id: The ID of the thread to delete

    Returns:
        Success message or 404 if thread not found
    """
    success = await ThreadService.delete_thread(thread_id)
    if success:
        return Success(success=True, message="Thread deleted successfully")
    return Error(message="Thread not found")


class DeleteAllThreadsResponse(Schema):
    success: bool
    deleted_count: int


@router.delete("/threads/", response={200: DeleteAllThreadsResponse})
async def delete_all_threads(request: HttpRequest) -> DeleteAllThreadsResponse:
    """
    Delete all conversation threads and their messages.

    Returns:
        Number of threads deleted
    """
    deleted_count = await ThreadService.delete_all_threads()
    return DeleteAllThreadsResponse(success=True, deleted_count=deleted_count)


# ============================================================================
# Message Management Endpoints
# ============================================================================


@router.post(
    "/threads/{thread_id}/messages/{message_id}/rate/",
    response={200: MessageResponse, 404: Error},
)
async def rate_message(
    request: HttpRequest,
    thread_id: str,
    message_id: str,
    payload: RateMessagePayload,
) -> MessageResponse | Error:
    """
    Rate a message as good (1) or bad (-1).

    Args:
        thread_id: The thread containing the message
        message_id: The UUID of the message to rate
        payload: Rating payload with value 1 (good) or -1 (bad)

    Returns:
        The updated message data
    """
    # Find thread and assistant
    thread = await ThreadService.get_assistant(thread_id)
    if not thread:
        return Error(message="Thread not found")

    assistant_id = thread.assistant_id
    assistant = registry.get(assistant_id)
    if not assistant:
        return Error(message=f"Assistant '{assistant_id}' not found")

    storage = await assistant.get_storage_adapter(thread_id)
    if not storage:
        return Error(message="Storage not available")

    # Rate the message
    success = await storage.rate_message(message_id, payload.rating)
    if not success:
        return Error(message="Message not found")

    return MessageResponse(id=message_id, rating=payload.rating)


@router.post(
    "/threads/{thread_id}/messages/{message_id}/delete/",
    response={200: MessageResponse, 404: Error},
)
async def delete_message(
    request: HttpRequest, thread_id: str, message_id: str
) -> MessageResponse | Error:
    """
    Soft delete a message.

    Args:
        thread_id: The thread containing the message
        message_id: The UUID of the message to delete

    Returns:
        Confirmation of deletion
    """
    # Find thread and assistant
    thread = await ThreadService.get_assistant(thread_id)
    if not thread:
        return Error(message="Thread not found")

    assistant_id = thread.assistant_id
    if assistant_id not in registry:
        return Error(message=f"Assistant '{assistant_id}' not found")

    assistant = registry.get(assistant_id)
    if not assistant:
        return Error(message=f"Assistant '{assistant_id}' not found")

    storage = await assistant.get_storage_adapter(thread_id)
    if not storage:
        return Error(message="Storage not available")

    # Delete the message
    success = await storage.delete_message(message_id)
    if not success:
        return Error(message="Message not found")

    return MessageResponse(id=message_id, is_deleted=True)


@router.post(
    "/threads/{thread_id}/messages/{message_id}/restore/",
    response={200: MessageResponse, 404: Error},
)
async def restore_message(
    request: HttpRequest, thread_id: str, message_id: str
) -> MessageResponse | Error:
    """
    Restore a soft-deleted message.

    Args:
        thread_id: The thread containing the message
        message_id: The UUID of the message to restore

    Returns:
        The restored message data
    """
    # Find thread and assistant
    thread = await ThreadService.get_assistant(thread_id)
    if not thread:
        return Error(message="Thread not found")

    assistant_id = thread.assistant_id
    if assistant_id not in registry:
        return Error(message=f"Assistant '{assistant_id}' not found")

    assistant = registry.get(assistant_id)
    if not assistant:
        return Error(message=f"Assistant '{assistant_id}' not found")

    storage = await assistant.get_storage_adapter(thread_id)
    if not storage:
        return Error(message="Storage not available")

    # Restore the message
    success = await storage.restore_message(message_id)
    if not success:
        return Error(message="Message not found")

    return MessageResponse(id=message_id, is_deleted=False)
