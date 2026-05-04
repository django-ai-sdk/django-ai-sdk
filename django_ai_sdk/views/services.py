from typing import Any, TypedDict

from asgiref.sync import async_to_sync
from django.db.models import Count
from django.http import HttpRequest, StreamingHttpResponse

from django_ai_sdk import Assistant
from django_ai_sdk.assistants import AssistantInfo
from django_ai_sdk.assistants.registry import registry
from django_ai_sdk.conversation.models import Thread
from django_ai_sdk.conversation.utils import generate_thread_title
from django_ai_sdk.memories.models import Entry, ThreadMemory
from django_ai_sdk.storage import ThreadService
from django_ai_sdk.views.schemas import ChatRequest


class ThreadInfo(TypedDict):
    id: str
    title: str
    assistant_id: str
    created_at: str
    updated_at: str
    message_count: int


class MemoryInfo(TypedDict):
    id: str
    name: str
    description: str
    document_count: int
    active: bool


class ThreadHistory(TypedDict):
    thread: dict[str, Any]
    memories: list[MemoryInfo]
    messages: list[Any]
    file_count: int
    file_memory_id: str | None


class MessageRating(TypedDict):
    id: str
    rating: int
    is_deleted: bool


class MessageStatus(TypedDict):
    id: str
    is_deleted: bool


class AssistantSummary(TypedDict):
    id: str
    name: str | None
    model: str | None


class ReindexResult(TypedDict):
    success: bool
    message: str


async def get_assistant(thread_id: str) -> Assistant:
    thread = await ThreadService.get_assistant(thread_id)
    if thread is None:
        raise ValueError("Thread not found")
    assistant = registry.get(thread.assistant_id)
    if assistant is None:
        raise ValueError(f"Assistant '{thread.assistant_id}' not found")
    return assistant


async def _list_threads(user_id: str) -> list[ThreadInfo]:
    threads = await ThreadService.threads(user_id=user_id)
    return [
        ThreadInfo(
            id=t.id,
            title=t.title,
            assistant_id=t.assistant_id,
            created_at=t.created_at.isoformat(),
            updated_at=t.updated_at.isoformat(),
            message_count=t.message_count,
        )
        for t in threads
    ]


alist_threads = _list_threads
list_threads = async_to_sync(_list_threads)


async def _create_thread(chat_request: ChatRequest, user_id: str) -> str:
    assistant_id = chat_request.assistant_id
    if assistant_id is None:
        raise ValueError("assistant_id is required")

    assistant = registry.get(assistant_id)
    if assistant is None:
        raise ValueError(f"Assistant '{assistant_id}' not found")

    title = generate_thread_title(chat_request.messages)

    thread_id = await ThreadService.create_thread(
        assistant_id=assistant_id,
        title=title,
        metadata={
            "model": assistant.model,
            "assistant_name": assistant.name or assistant.__class__.__name__,
            "assistant_class": assistant.__class__.__name__,
            "created_via": "create_thread",
        },
        user_id=user_id,
    )

    return thread_id


acreate_thread = _create_thread
create_thread = async_to_sync(_create_thread)


async def _get_thread_history(thread_id: str) -> ThreadHistory:
    assistant = await get_assistant(thread_id)
    thread_detail = await assistant.history(thread_id)
    thread = await Thread.objects.select_related("file_memory").aget(id=thread_id)

    thread_memories = (
        ThreadMemory.objects.filter(thread_id=thread_id, memory__is_hidden=False)
        .select_related("memory")
        .annotate(document_count=Count("memory__entries"))
    )

    memories: list[MemoryInfo] = []
    async for tm in thread_memories:
        memories.append(
            MemoryInfo(
                id=str(tm.memory.id),
                name=tm.memory.name,
                description=tm.memory.description,
                document_count=tm.document_count,
                active=tm.active,
            )
        )

    file_memory_id = str(thread.file_memory_id) if thread.file_memory_id else None
    file_count = (
        await Entry.objects.filter(memory_id=thread.file_memory_id).acount()
        if file_memory_id
        else 0
    )

    return ThreadHistory(
        thread=thread_detail.thread,
        memories=memories,
        messages=thread_detail.messages,
        file_count=file_count,
        file_memory_id=file_memory_id,
    )


aget_thread_history = _get_thread_history
get_thread_history = async_to_sync(_get_thread_history)


async def _add_message_to_thread(
    thread_id: str,
    messages: list[dict[str, Any]],
    request: HttpRequest,
) -> StreamingHttpResponse:
    assistant = await get_assistant(thread_id)
    return await assistant.as_view(messages, thread_id=thread_id)


aadd_message_to_thread = _add_message_to_thread


async def _delete_thread(thread_id: str) -> bool:
    return await ThreadService.delete_thread(thread_id)


adelete_thread = _delete_thread
delete_thread = async_to_sync(_delete_thread)


async def _delete_all_threads() -> int:
    return await ThreadService.delete_all_threads()


adelete_all_threads = _delete_all_threads
delete_all_threads = async_to_sync(_delete_all_threads)


async def _rate_message(thread_id: str, message_id: str, rating: int) -> MessageRating:
    assistant = await get_assistant(thread_id)
    storage = await assistant.get_storage_adapter(thread_id)
    if not storage:
        raise ValueError("Storage not found")

    success = await storage.rate_message(message_id, rating)
    if not success:
        raise ValueError("Message not found")
    return MessageRating(id=message_id, rating=rating, is_deleted=False)


arate_message = _rate_message
rate_message = async_to_sync(_rate_message)


async def _delete_message(thread_id: str, message_id: str) -> MessageStatus:
    assistant = await get_assistant(thread_id)
    storage = await assistant.get_storage_adapter(thread_id)
    if not storage:
        raise ValueError("Storage not found")

    success = await storage.delete_message(message_id)
    if not success:
        raise ValueError("Message not found")
    return MessageStatus(id=message_id, is_deleted=True)


adelete_message = _delete_message
delete_message = async_to_sync(_delete_message)


async def _restore_message(thread_id: str, message_id: str) -> MessageStatus:
    assistant = await get_assistant(thread_id)
    storage = await assistant.get_storage_adapter(thread_id)
    if not storage:
        raise ValueError("Storage not found")

    success = await storage.restore_message(message_id)
    if not success:
        raise ValueError("Message not found")
    return MessageStatus(id=message_id, is_deleted=False)


arestore_message = _restore_message
restore_message = async_to_sync(_restore_message)


async def _list_assistants() -> list[AssistantSummary]:
    return [
        AssistantSummary(
            id=assistant_id,
            name=assistant.name,
            model=assistant.model,
        )
        for assistant_id, assistant in registry.all().items()
    ]


alist_assistants = _list_assistants
list_assistants = async_to_sync(_list_assistants)


async def _get_assistant_info(assistant_id: str) -> AssistantInfo:
    assistant = registry.get(assistant_id)
    if assistant is None:
        raise ValueError(f"Assistant '{assistant_id}' not found")
    return assistant.info()


aget_assistant_info = _get_assistant_info
get_assistant_info = async_to_sync(_get_assistant_info)


async def _reindex_assistant(
    assistant_id: str, memory_id: str | None = None, force_rebuild: bool = False
) -> ReindexResult:
    assistant = registry.get(assistant_id)
    if assistant is None:
        raise ValueError(f"Assistant '{assistant_id}' not found")

    result = await Assistant.reindex(assistant, memory_id, force_rebuild)

    if not result:
        return ReindexResult(
            success=False, message="No RAG provider configured for this assistant"
        )

    rebuild_msg = " (force rebuild)" if force_rebuild else ""
    return ReindexResult(
        success=True,
        message="RAG pipeline reindexed successfully"
        + rebuild_msg
        + (f" for memory {memory_id}" if memory_id else ""),
    )


areindex_assistant = _reindex_assistant
reindex_assistant = async_to_sync(_reindex_assistant)
