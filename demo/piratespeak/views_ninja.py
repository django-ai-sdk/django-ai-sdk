from typing import Any

from django.http import HttpRequest
from django_ai_sdk import Assistant
from django_ai_sdk.assistants import AssistantInfo
from django_ai_sdk.assistants.services import AssistantService
from django_ai_sdk.storage.services import (
    ThreadService,
    aget_thread_file_meta,
    aget_thread_history,
)
from django_ai_sdk.views.schemas import ChatRequest, RateMessagePayload
from ninja import Router, Schema

router = Router()


@router.get("/health/")
def health_check(request: HttpRequest) -> dict:
    return {"status": "ok", "service": "piratespeak"}


class Error(Schema):
    message: str
    code: int | None = None


class Success(Schema):
    success: bool
    message: str | None = None


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


class ThreadDetailResponse(Schema):
    thread: dict
    messages: list


class ThreadFileMeta(Schema):
    file_count: int = 0
    file_memory_id: str | None = None


class DeleteAllThreadsResponse(Schema):
    success: bool
    deleted_count: int


class MessageResponse(Schema):
    id: str
    rating: int | None = None
    is_deleted: bool = False


@router.get("/threads/", response={200: ThreadListResponse, 500: Error})
async def list_threads(request: HttpRequest) -> Any:
    try:
        all_threads = await ThreadService.threads(user_id=request.user.id)
        items = [
            ThreadListItem(
                id=t.id,
                title=t.title,
                assistant_id=t.assistant_id,
                created_at=t.created_at.isoformat(),
                updated_at=t.updated_at.isoformat(),
                message_count=t.message_count,
            )
            for t in all_threads
        ]
        return ThreadListResponse(threads=items)
    except Exception as e:
        return 500, Error(message=str(e))


@router.post("/threads/", response={200: CreateThreadResponse, 400: Error})
async def create_thread(request: HttpRequest, payload: ChatRequest) -> Any:
    try:
        thread_id = await ThreadService.create_thread(
            assistant_id=payload.assistant_id or "",
            messages=payload.messages,
            user_id=request.user.id,
        )
        return CreateThreadResponse(thread_id=thread_id)
    except ValueError as e:
        return 400, Error(message=str(e))
    except Exception as e:
        return 500, Error(message=str(e))


@router.get("/threads/{thread_id}/", response={200: ThreadDetailResponse, 404: Error})
async def get_thread_history(request: HttpRequest, thread_id: str) -> Any:
    try:
        data = await aget_thread_history(thread_id)
        return ThreadDetailResponse(**data)
    except ValueError as e:
        return 404, Error(message=str(e))


@router.get("/threads/{thread_id}/file-meta/", response={200: ThreadFileMeta, 404: Error})
async def get_thread_file_meta(request: HttpRequest, thread_id: str) -> Any:
    try:
        data = await aget_thread_file_meta(thread_id)
        return ThreadFileMeta(**data)
    except ValueError as e:
        return 404, Error(message=str(e))


@router.post("/threads/{thread_id}/", response={404: Error, 500: Error})
async def add_message_to_thread(request: HttpRequest, thread_id: str, payload: ChatRequest) -> Any:
    try:
        assistant = await AssistantService.get_assistant(thread_id)
        return await assistant.as_view(payload.messages, thread_id=thread_id)
    except ValueError as e:
        return 404, Error(message=str(e))


@router.delete("/threads/{thread_id}/", response={200: Success, 404: Error})
async def delete_thread(request: HttpRequest, thread_id: str) -> Any:
    try:
        success = await ThreadService.delete_thread(thread_id)
        if success:
            return Success(success=True, message="Thread deleted successfully")
        return 404, Error(message="Thread not found")
    except ValueError as e:
        return 404, Error(message=str(e))
    except Exception as e:
        return 500, Error(message=str(e))


@router.delete("/threads/", response={200: DeleteAllThreadsResponse})
async def delete_all_threads(request: HttpRequest) -> Any:
    try:
        deleted_count = await ThreadService.delete_all_threads()
        return DeleteAllThreadsResponse(success=True, deleted_count=deleted_count)
    except Exception as e:
        return 500, Error(message=str(e))


@router.post(
    "/threads/{thread_id}/messages/{message_id}/rate/",
    response={200: MessageResponse, 404: Error},
)
async def rate_message(
    request: HttpRequest,
    thread_id: str,
    message_id: str,
    payload: RateMessagePayload,
) -> Any:
    try:
        await ThreadService.rate_message(thread_id, message_id, payload.rating)
        return MessageResponse(id=message_id, rating=payload.rating, is_deleted=False)
    except ValueError as e:
        return 404, Error(message=str(e))


@router.post(
    "/threads/{thread_id}/messages/{message_id}/delete/",
    response={200: MessageResponse, 404: Error},
)
async def delete_message(request: HttpRequest, thread_id: str, message_id: str) -> Any:
    try:
        await ThreadService.delete_message(thread_id, message_id)
        return MessageResponse(id=message_id, is_deleted=True)
    except ValueError as e:
        return 404, Error(message=str(e))


@router.post(
    "/threads/{thread_id}/messages/{message_id}/restore/",
    response={200: MessageResponse, 404: Error},
)
async def restore_message(request: HttpRequest, thread_id: str, message_id: str) -> Any:
    try:
        await ThreadService.restore_message(thread_id, message_id)
        return MessageResponse(id=message_id, is_deleted=False)
    except ValueError as e:
        return 404, Error(message=str(e))


@router.get("/assistants/")
async def list_assistants_view(request: HttpRequest) -> Any:
    try:
        items = AssistantService.list_assistants()
        return {"assistants": items}
    except Exception as e:
        return 500, Error(message=str(e))


@router.get("/assistants/{assistant_id}/", response={200: AssistantInfo, 404: Error})
async def get_assistant_info(request: HttpRequest, assistant_id: str) -> Any:
    try:
        assistant = AssistantService.from_registry(assistant_id)
        return assistant.info()
    except ValueError as e:
        return 404, Error(message=str(e))


@router.post(
    "/assistants/{assistant_id}/reindex/",
    response={200: Success, 404: Error},
)
async def reindex_assistant(
    request: HttpRequest,
    assistant_id: str,
    memory_id: str | None = None,
    force_rebuild: bool = False,
) -> Any:
    try:
        assistant = AssistantService.from_registry(assistant_id)
        result = await Assistant.reindex(assistant, memory_id, force_rebuild)

        if not result:
            return Success(success=False, message="No RAG provider configured for this assistant")

        rebuild_msg = " (force rebuild)" if force_rebuild else ""
        message = "RAG pipeline reindexed successfully" + rebuild_msg
        if memory_id:
            message += f" for memory {memory_id}"

        return Success(success=True, message=message)
    except ValueError as e:
        return 404, Error(message=str(e))
    except Exception as e:
        return 500, Error(message=str(e))
