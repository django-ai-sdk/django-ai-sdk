from typing import Any

from django.http import HttpRequest
from django_ai_sdk import Assistant
from django_ai_sdk.assistants import AssistantInfo
from django_ai_sdk.assistants.services import AssistantService
from django_ai_sdk.memories.services import MemoryService
from django_ai_sdk.permissions import PermissionDenied
from django_ai_sdk.storage.schemas import ThreadInfo
from django_ai_sdk.storage.services import (
    ThreadService,
    aget_thread_file_meta,
    aget_thread_history,
)
from django_ai_sdk.views.schemas import ChatRequest, RateMessagePayload
from ninja import Router, Schema

router = Router()


class Error(Schema):
    message: str
    code: int | None = None


class Success(Schema):
    success: bool
    message: str | None = None


class HealthResponse(Schema):
    status: str
    service: str


class AssistantItem(Schema):
    id: str
    name: str | None = None
    model: str | None = None


class AssistantsListResponse(Schema):
    assistants: list[AssistantItem]


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
    thread: ThreadInfo
    messages: list


class ThreadFileMeta(Schema):
    file_count: int = 0
    file_memory_id: str | None = None


class DeleteAllThreadsResponse(Schema):
    success: bool
    deleted_count: int


class FeedbackResponse(Schema):
    id: str
    user_id: str | None = None
    rating: int
    feedback: str
    created_at: str | None = None


class PatchThreadPayload(Schema):
    assistant_id: str


class MessageResponse(Schema):
    id: str
    is_deleted: bool = False


@router.get("/health/", response={200: HealthResponse})
def health_check(request: HttpRequest) -> HealthResponse:
    return HealthResponse(status="ok", service="piratespeak")


@router.get("/threads/", response={200: ThreadListResponse, 500: Error})
async def list_threads(request: HttpRequest) -> Any:
    try:
        all_threads = await ThreadService.threads(user=request.user)
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


@router.post("/threads/", response={200: CreateThreadResponse, 400: Error, 500: Error})
async def create_thread(request: HttpRequest, payload: ChatRequest) -> Any:
    try:
        assistant_id = payload.assistant_id or ""
        thread_id = await ThreadService.create_thread(
            assistant_id=assistant_id,
            messages=payload.messages,
            user=request.user,
        )
        await MemoryService.link_memories(assistant_id, thread_id, user=request.user)
        return CreateThreadResponse(thread_id=thread_id)
    except ValueError as e:
        return 400, Error(message=str(e))
    except Exception as e:
        return 500, Error(message=str(e))


@router.get("/threads/{thread_id}/", response={200: ThreadDetailResponse, 404: Error, 403: Error})
async def get_thread_history(request: HttpRequest, thread_id: str) -> Any:
    try:
        data = await aget_thread_history(thread_id, user=request.user)
        return ThreadDetailResponse(**data)
    except ValueError as e:
        return 404, Error(message=str(e))


@router.get("/threads/{thread_id}/file-meta/", response={200: ThreadFileMeta, 404: Error})
async def get_thread_file_meta(request: HttpRequest, thread_id: str) -> Any:
    try:
        data = await aget_thread_file_meta(thread_id, user=request.user)
        return ThreadFileMeta(**data)
    except ValueError as e:
        return 404, Error(message=str(e))


@router.post("/threads/{thread_id}/", response={403: Error, 404: Error, 500: Error})
async def add_message_to_thread(request: HttpRequest, thread_id: str, payload: ChatRequest) -> Any:
    try:
        assistant = await AssistantService.get_assistant(thread_id, user=request.user)
        return await assistant.as_view(payload.messages, thread_id=thread_id, user=request.user)
    except PermissionDenied as e:
        return 403, Error(message=str(e))
    except ValueError as e:
        return 404, Error(message=str(e))


@router.delete("/threads/{thread_id}/", response={200: Success, 403: Error, 404: Error, 500: Error})
async def delete_thread(request: HttpRequest, thread_id: str) -> Any:
    try:
        success = await ThreadService.delete_thread(thread_id, user=request.user)
        if success:
            return Success(success=True, message="Thread deleted successfully")
        return 404, Error(message="Thread not found")
    except PermissionDenied as e:
        return 403, Error(message=str(e))
    except ValueError as e:
        return 404, Error(message=str(e))
    except Exception as e:
        return 500, Error(message=str(e))


@router.delete("/threads/", response={200: DeleteAllThreadsResponse, 403: Error, 500: Error})
async def delete_all_threads(request: HttpRequest) -> Any:
    try:
        deleted_count = await ThreadService.delete_all_threads(user=request.user)
        return DeleteAllThreadsResponse(success=True, deleted_count=deleted_count)
    except PermissionDenied as e:
        return 403, Error(message=str(e))
    except Exception as e:
        return 500, Error(message=str(e))


@router.patch("/threads/{thread_id}/", response={200: Success, 400: Error, 403: Error, 404: Error})
async def patch_thread(request: HttpRequest, thread_id: str, payload: PatchThreadPayload) -> Any:
    try:
        AssistantService.from_registry(payload.assistant_id)
        thread = await ThreadService.get_thread(thread_id, user=request.user)
        if thread is None:
            return 404, Error(message="Thread not found")
        if thread.assistant_id:
            await MemoryService.unlink_memories(thread.assistant_id, thread_id, user=request.user)
        await ThreadService.update_thread(
            thread_id, metadata={"assistant_id": payload.assistant_id}, user=request.user
        )
        await MemoryService.link_memories(payload.assistant_id, thread_id, user=request.user)
        return Success(success=True)
    except PermissionDenied as e:
        return 403, Error(message=str(e))
    except ValueError as e:
        return 400, Error(message=str(e))


@router.post(
    "/threads/{thread_id}/messages/{message_id}/rate/",
    response={200: MessageResponse, 403: Error, 404: Error},
)
async def rate_message(
    request: HttpRequest,
    thread_id: str,
    message_id: str,
    payload: RateMessagePayload,
) -> Any:

    try:
        await ThreadService.rate_message(
            thread_id, message_id, payload.rating, feedback=payload.feedback, user=request.user
        )
        return MessageResponse(id=message_id, is_deleted=False)

    except PermissionDenied as e:
        return 403, Error(message=str(e))
    except ValueError as e:
        return 404, Error(message=str(e))


@router.post(
    "/threads/{thread_id}/messages/{message_id}/delete/",
    response={200: MessageResponse, 403: Error, 404: Error},
)
async def delete_message(request: HttpRequest, thread_id: str, message_id: str) -> Any:
    try:
        await ThreadService.delete_message(thread_id, message_id, user=request.user)
        return MessageResponse(id=message_id, is_deleted=True)
    except PermissionDenied as e:
        return 403, Error(message=str(e))
    except ValueError as e:
        return 404, Error(message=str(e))


@router.post(
    "/threads/{thread_id}/messages/{message_id}/restore/",
    response={200: MessageResponse, 403: Error, 404: Error},
)
async def restore_message(request: HttpRequest, thread_id: str, message_id: str) -> Any:
    try:
        await ThreadService.restore_message(thread_id, message_id, user=request.user)
        return MessageResponse(id=message_id, is_deleted=False)
    except PermissionDenied as e:
        return 403, Error(message=str(e))
    except ValueError as e:
        return 404, Error(message=str(e))


@router.get("/assistants/", response={200: AssistantsListResponse, 403: Error, 500: Error})
async def list_assistants(request: HttpRequest) -> Any:
    try:
        items = await AssistantService.list_assistants(user=request.user)
        return AssistantsListResponse(assistants=[AssistantItem(**item) for item in items])
    except PermissionDenied as e:
        return 403, Error(message=str(e))
    except Exception as e:
        return 500, Error(message=str(e))


@router.get("/assistants/{assistant_id}/", response={200: AssistantInfo, 403: Error, 404: Error})
async def get_assistant_info(request: HttpRequest, assistant_id: str) -> Any:
    try:
        return await AssistantService.get_assistant_info(assistant_id, user=request.user)
    except PermissionDenied as e:
        return 403, Error(message=str(e))
    except ValueError as e:
        return 404, Error(message=str(e))


@router.post(
    "/assistants/{assistant_id}/reindex/",
    response={200: Success, 404: Error, 500: Error},
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
