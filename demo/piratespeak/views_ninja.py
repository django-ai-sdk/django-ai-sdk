from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from django.http import HttpRequest
from django_ai_sdk import Assistant
from django_ai_sdk.assistants.services import AssistantService
from django_ai_sdk.logger import get_logger
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
logger = get_logger(__name__)


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
    file_upload: bool = False
    rag: bool = False


class AssistantsListResponse(Schema):
    assistants: list[AssistantItem]


class AssistantInfoResponse(Schema):
    id: str
    name: str | None = None
    model: str | None = None
    class_name: str
    description: str | None = None
    instructions: str | None = None
    file_upload: bool = False
    rag: bool = False


class Tool(Schema):
    label: str
    description: str | None = None
    children: list[Tool] = []


class IntegrationStatusOut(Schema):
    server_name: str
    label: str
    type: str
    status: str
    tool_names: list[str]


class ToolsResponse(Schema):
    tools: list[Tool]
    integrations: list[IntegrationStatusOut] = []


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


class FeedbackResponse(Schema):
    id: str
    user_id: str | None = None
    rating: int
    feedback: str
    created_at: str | None = None


class ThreadMessage(Schema):
    id: str
    role: str
    parts: list = []
    finish_reason: str | None = None
    tool_calls: list = []
    processing_time_ms: int | None = None
    has_errors: bool = False
    usage: dict | None = None
    feedback: FeedbackResponse | None = None
    created_at: str | None = None


class ThreadDetailResponse(Schema):
    thread: ThreadInfo
    messages: list[ThreadMessage]


class ThreadFileMeta(Schema):
    file_count: int = 0
    file_memory_id: str | None = None


class DeleteAllThreadsResponse(Schema):
    success: bool
    deleted_count: int


class PatchThreadPayload(Schema):
    assistant_id: str


class MessageResponse(Schema):
    id: str
    is_deleted: bool = False
    feedback: FeedbackResponse | None = None


class RunResponse(Schema):
    result: str | None = None
    thread_id: str


@router.get("/health/", response={200: HealthResponse}, operation_id="health_check")
def health_check(request: HttpRequest) -> HealthResponse:
    return HealthResponse(status="ok", service="piratespeak")


@router.get(
    "/threads/", response={200: ThreadListResponse, 500: Error}, operation_id="list_threads"
)
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


@router.post(
    "/threads/",
    response={200: CreateThreadResponse, 400: Error, 500: Error},
    operation_id="create_thread",
)
async def create_thread(request: HttpRequest, payload: ChatRequest) -> Any:
    try:
        assistant_id = payload.assistant_id or ""
        # Initial messages are not persisted here; the chat/stream endpoint
        # receives and stores the full message list.
        thread_id = await ThreadService.create_thread(
            assistant_id=assistant_id,
            user=request.user,
        )
        await MemoryService.link_memories(assistant_id, thread_id, user=request.user)
        return CreateThreadResponse(thread_id=thread_id)
    except ValueError as e:
        return 400, Error(message=str(e))
    except Exception as e:
        return 500, Error(message=str(e))


@router.get(
    "/threads/{thread_id}/",
    response={200: ThreadDetailResponse, 404: Error, 403: Error},
    operation_id="get_thread_history",
)
async def get_thread_history(request: HttpRequest, thread_id: str) -> Any:
    try:
        data = await aget_thread_history(thread_id, user=request.user)

        # Filter feedbacks to current user only
        user_pk = str(request.user.pk) if request.user.is_authenticated else None
        for message in data.get("messages", []):
            feedbacks = message.get("feedbacks", [])
            user_feedback = None
            if feedbacks:
                user_feedback = next(
                    (fb for fb in feedbacks if fb.get("user_id") == user_pk),
                    None,
                )
            message["feedback"] = user_feedback
            del message["feedbacks"]

        return ThreadDetailResponse(**data)
    except PermissionDenied as e:
        return 403, Error(message=str(e))
    except ValueError as e:
        return 404, Error(message=str(e))


@router.get(
    "/threads/{thread_id}/file-meta/",
    response={200: ThreadFileMeta, 404: Error},
    operation_id="get_thread_file_meta",
)
async def get_thread_file_meta(request: HttpRequest, thread_id: str) -> Any:
    try:
        data = await aget_thread_file_meta(thread_id, user=request.user)
        return ThreadFileMeta(**data)
    except ValueError as e:
        return 404, Error(message=str(e))


@router.post(
    "/assistants/{assistant_id}/run/",
    response={200: RunResponse, 400: Error, 403: Error, 404: Error, 500: Error, 501: Error},
    operation_id="run_assistant",
)
async def run_assistant(request: HttpRequest, assistant_id: str, payload: ChatRequest) -> Any:
    try:
        assistant = await AssistantService.get(assistant_id)
        chat_messages = assistant.protocol_handler.to_chat_messages(payload.messages)
        result = await assistant.run(chat_messages, user=request.user)
        return RunResponse(result=result, thread_id="")
    except PermissionDenied as e:
        return 403, Error(message=str(e))
    except ValueError as e:
        return 404, Error(message=str(e))
    except NotImplementedError as e:
        return 501, Error(message=str(e))
    except Exception as e:
        return 500, Error(message=str(e))


@router.post(
    "/threads/{thread_id}/run/",
    response={200: RunResponse, 400: Error, 403: Error, 404: Error, 500: Error, 501: Error},
    operation_id="run_thread",
)
async def run_thread(request: HttpRequest, thread_id: str, payload: ChatRequest) -> Any:
    try:
        assistant = await AssistantService.get_assistant(thread_id, user=request.user)
        chat_messages = assistant.protocol_handler.to_chat_messages(payload.messages)
        result = await assistant.run(chat_messages, thread_id=thread_id, user=request.user)
        return RunResponse(result=result, thread_id=thread_id)
    except PermissionDenied as e:
        return 403, Error(message=str(e))
    except ValueError as e:
        return 404, Error(message=str(e))
    except NotImplementedError as e:
        return 501, Error(message=str(e))
    except Exception as e:
        return 500, Error(message=str(e))


@router.post(
    "/threads/{thread_id}/",
    response={403: Error, 404: Error, 500: Error},
    operation_id="add_message_to_thread",
)
async def add_message_to_thread(request: HttpRequest, thread_id: str, payload: ChatRequest) -> Any:
    try:
        assistant = await AssistantService.get_assistant(thread_id, user=request.user)
        return await assistant.as_view(payload.messages, thread_id=thread_id, user=request.user)
    except PermissionDenied as e:
        return 403, Error(message=str(e))
    except ValueError as e:
        return 404, Error(message=str(e))


@router.delete(
    "/threads/{thread_id}/",
    response={200: Success, 403: Error, 404: Error, 500: Error},
    operation_id="delete_thread",
)
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


@router.delete(
    "/threads/",
    response={200: DeleteAllThreadsResponse, 403: Error, 500: Error},
    operation_id="delete_all_threads",
)
async def delete_all_threads(request: HttpRequest) -> Any:
    try:
        deleted_count = await ThreadService.delete_all_threads(user=request.user)
        return DeleteAllThreadsResponse(success=True, deleted_count=deleted_count)
    except PermissionDenied as e:
        return 403, Error(message=str(e))
    except Exception as e:
        return 500, Error(message=str(e))


@router.patch(
    "/threads/{thread_id}/",
    response={200: Success, 400: Error, 403: Error, 404: Error},
    operation_id="patch_thread",
)
async def patch_thread(request: HttpRequest, thread_id: str, payload: PatchThreadPayload) -> Any:
    try:
        await AssistantService.get(payload.assistant_id)
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
    operation_id="rate_message",
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
    operation_id="delete_message",
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
    operation_id="restore_message",
)
async def restore_message(request: HttpRequest, thread_id: str, message_id: str) -> Any:
    try:
        await ThreadService.restore_message(thread_id, message_id, user=request.user)
        return MessageResponse(id=message_id, is_deleted=False)
    except PermissionDenied as e:
        return 403, Error(message=str(e))
    except ValueError as e:
        return 404, Error(message=str(e))


@router.get(
    "/assistants/",
    response={200: AssistantsListResponse, 403: Error, 500: Error},
    operation_id="list_assistants",
)
async def list_assistants(request: HttpRequest) -> Any:
    try:
        items = await AssistantService.list_assistants(user=request.user)
        return AssistantsListResponse(assistants=[AssistantItem(**item) for item in items])
    except PermissionDenied as e:
        return 403, Error(message=str(e))
    except Exception as e:
        return 500, Error(message=str(e))


@router.get(
    "/assistants/{assistant_id}/",
    response={200: AssistantInfoResponse, 403: Error, 404: Error},
    operation_id="get_assistant_info",
)
async def get_assistant_info(request: HttpRequest, assistant_id: str) -> Any:
    try:
        assistant = await AssistantService.get(assistant_id)
        info = await AssistantService.get_assistant_info(assistant_id, user=request.user)
        return AssistantInfoResponse(
            id=info.id,
            name=info.name,
            model=info.model,
            class_name=info.class_name,
            description=info.description,
            instructions=assistant.get_system_prompt(),
            file_upload=info.file_upload,
            rag=info.rag,
        )
    except PermissionDenied as e:
        return 403, Error(message=str(e))
    except ValueError as e:
        return 404, Error(message=str(e))


@router.get(
    "/assistants/{assistant_id}/tools/",
    response={200: ToolsResponse, 404: Error},
    operation_id="get_assistant_tools",
)
async def get_assistant_tools(request: HttpRequest, assistant_id: str) -> Any:
    try:
        assistant = await AssistantService.get(assistant_id)
    except ValueError as e:
        return 404, Error(message=str(e))

    tools_data = []
    try:
        tool_objs = await assistant.get_tools()
        tools_data = [
            Tool(
                label=getattr(t, "label", None) or t.name.replace("_", " ").title(),
                description=t.description or "",
            )
            for t in tool_objs
        ]
    except Exception:
        logger.exception("Failed to build tools for assistant %s", assistant_id)

    integrations_data = []
    try:
        integration_status = await AssistantService.get_integration_status(
            assistant, user=request.user
        )
        integrations_data = [
            IntegrationStatusOut(
                server_name=s.server_name,
                label=s.label,
                type=s.type,
                status=s.status,
                tool_names=s.tool_names,
            )
            for s in integration_status
        ]
    except Exception:
        logger.exception("Failed to load integration status for assistant %s", assistant_id)

    return ToolsResponse(tools=tools_data, integrations=integrations_data)


@router.post(
    "/assistants/{assistant_id}/reindex/",
    response={200: Success, 404: Error, 500: Error},
    operation_id="reindex_assistant",
)
async def reindex_assistant(
    request: HttpRequest,
    assistant_id: str,
    memory_id: str | None = None,
    force_rebuild: bool = False,
) -> Any:
    try:
        assistant = await AssistantService.get(assistant_id)
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


# ============================================================================
# Runtime Assistants (DB-configured)
# ============================================================================


class AssistantSettingsOut(Schema):
    id: UUID
    name: str
    slug: str
    assistant: str
    model: str
    system_prompt: str
    tools: list[str]
    integrations: list[str]
    memories: list[str]
    suggestion_enabled: bool
    title_generation: bool
    max_history: int | None
    file_upload: bool
    active: bool
    created_at: datetime
    updated_at: datetime


class AssistantSettingsCreateUserEntry(Schema):
    user_id: str
    can_manage: bool = False


class AssistantSettingsCreateGroupEntry(Schema):
    group_id: int
    can_manage: bool = False


class AssistantSettingsCreateIn(Schema):
    name: str
    slug: str = ""
    assistant: str = ""
    model: str = "gpt-4o"
    system_prompt: str = ""
    tools: list[str] = []
    integrations: list[str] = []
    memories: list[str] = []
    users: list[AssistantSettingsCreateUserEntry] = []
    groups: list[AssistantSettingsCreateGroupEntry] = []
    suggestion_enabled: bool = False
    title_generation: bool = True
    max_history: int | None = None
    file_upload: bool = False


class AssistantSettingsUpdateIn(Schema):
    name: str | None = None
    assistant: str | None = None
    model: str | None = None
    system_prompt: str | None = None
    tools: list[str] | None = None
    integrations: list[str] | None = None
    memories: list[str] | None = None
    suggestion_enabled: bool | None = None
    title_generation: bool | None = None
    max_history: int | None = None
    file_upload: bool | None = None
    active: bool | None = None


class RuntimeAssistantBaseItem(Schema):
    path: str
    name: str


class RuntimeAssistantToolItem(Schema):
    key: str
    path: str


@router.get(
    "/assistants/runtimes/bases/",
    response={200: list[RuntimeAssistantBaseItem]},
    operation_id="list_runtime_assistant_bases",
)
def list_runtime_assistant_bases(request: HttpRequest) -> list[RuntimeAssistantBaseItem]:
    from django_ai_sdk.assistants.config import get_runtime_assistant_bases

    return [
        RuntimeAssistantBaseItem(
            path=f"{cls.__module__}.{cls.__qualname__}",
            name=cls.__name__,
        )
        for cls in get_runtime_assistant_bases()
    ]


@router.get(
    "/assistants/runtimes/tools/",
    response={200: list[RuntimeAssistantToolItem]},
    operation_id="list_runtime_assistant_tools",
)
def list_runtime_assistant_tools(request: HttpRequest) -> list[RuntimeAssistantToolItem]:
    from django_ai_sdk.assistants.config import get_tool_registry

    return [
        RuntimeAssistantToolItem(key=key, path=path) for key, path in get_tool_registry().items()
    ]


@router.get(
    "/assistants/runtimes/",
    response={200: list[AssistantSettingsOut]},
    operation_id="list_runtime_assistants",
)
async def list_runtime_assistants(request: HttpRequest) -> Any:
    return await AssistantService.list_runtime_assistants(user=request.user)


@router.post(
    "/assistants/runtimes/",
    response={200: AssistantSettingsOut, 400: Error},
    operation_id="create_runtime_assistant",
)
async def create_runtime_assistant(request: HttpRequest, payload: AssistantSettingsCreateIn) -> Any:
    try:
        config = await AssistantService.create_runtime_assistant(
            {
                "name": payload.name,
                "slug": payload.slug,
                "assistant": payload.assistant,
                "model": payload.model,
                "system_prompt": payload.system_prompt,
                "tools": payload.tools,
                "integrations": payload.integrations,
                "memories": payload.memories,
                "suggestion_enabled": payload.suggestion_enabled,
                "title_generation": payload.title_generation,
                "max_history": payload.max_history,
                "file_upload": payload.file_upload,
            },
            user=request.user,
        )
        for entry in payload.users:
            try:
                await AssistantService.add_assistant_user(
                    str(config.id), entry.user_id, entry.can_manage, user=request.user
                )
            except Exception:
                pass
        for entry in payload.groups:
            try:
                await AssistantService.add_assistant_group(
                    str(config.id), entry.group_id, entry.can_manage, user=request.user
                )
            except Exception:
                pass
        return config
    except Exception as e:
        return 400, Error(message=str(e))


@router.get(
    "/assistants/runtimes/{runtime_id}/",
    response={200: AssistantSettingsOut, 404: Error},
    operation_id="get_runtime_assistant",
)
async def get_runtime_assistant(request: HttpRequest, runtime_id: UUID) -> Any:
    try:
        return await AssistantService.get_runtime_assistant(str(runtime_id), user=request.user)
    except ValueError as e:
        return 404, Error(message=str(e))


@router.patch(
    "/assistants/runtimes/{runtime_id}/",
    response={200: AssistantSettingsOut, 404: Error, 400: Error},
    operation_id="update_runtime_assistant",
)
async def update_runtime_assistant(
    request: HttpRequest, runtime_id: UUID, payload: AssistantSettingsUpdateIn
) -> Any:
    try:
        from typing import cast

        from django_ai_sdk.assistants.services import AssistantUpdateData

        data = cast(
            "AssistantUpdateData",
            {k: v for k, v in payload.model_dump().items() if v is not None},
        )
        return await AssistantService.update_runtime_assistant(
            str(runtime_id), data, user=request.user
        )
    except ValueError as e:
        return 404, Error(message=str(e))
    except Exception as e:
        return 400, Error(message=str(e))


@router.delete(
    "/assistants/runtimes/{runtime_id}/",
    response={200: AssistantSettingsOut, 404: Error},
    operation_id="delete_runtime_assistant",
)
async def delete_runtime_assistant(request: HttpRequest, runtime_id: UUID) -> Any:
    try:
        return await AssistantService.delete_runtime_assistant(str(runtime_id), user=request.user)
    except ValueError as e:
        return 404, Error(message=str(e))


# ── Assistant Users ───────────────────────────────────────────────────────────


class AssistantUserOut(Schema):
    user_id: str
    can_manage: bool
    created_at: str


class AddAssistantUserIn(Schema):
    user_id: str
    can_manage: bool = False


class UpdateAssistantUserIn(Schema):
    can_manage: bool


@router.get(
    "/assistants/runtimes/{runtime_id}/users/",
    response={200: list[AssistantUserOut], 403: Error, 404: Error},
    operation_id="list_assistant_users",
)
async def list_assistant_users(request: HttpRequest, runtime_id: UUID) -> Any:
    try:
        users = await AssistantService.list_assistant_users(str(runtime_id), user=request.user)
        return [
            AssistantUserOut(
                user_id=str(u.user_id),
                can_manage=u.can_manage,
                created_at=u.created_at.isoformat() if u.created_at else "",
            )
            for u in users
        ]
    except ValueError as e:
        return 404, Error(message=str(e))


@router.post(
    "/assistants/runtimes/{runtime_id}/users/",
    response={200: AssistantUserOut, 403: Error, 404: Error},
    operation_id="add_assistant_user",
)
async def add_assistant_user(
    request: HttpRequest, runtime_id: UUID, payload: AddAssistantUserIn
) -> Any:
    try:
        entry = await AssistantService.add_assistant_user(
            str(runtime_id),
            payload.user_id,
            payload.can_manage,
            user=request.user,
        )
        return AssistantUserOut(
            user_id=str(entry.user_id),
            can_manage=entry.can_manage,
            created_at=entry.created_at.isoformat() if entry.created_at else "",
        )
    except PermissionDenied as e:
        return 403, Error(message=str(e))
    except ValueError as e:
        return 404, Error(message=str(e))


@router.patch(
    "/assistants/runtimes/{runtime_id}/users/{user_id}/",
    response={200: AssistantUserOut, 403: Error, 404: Error},
    operation_id="update_assistant_user",
)
async def update_assistant_user(
    request: HttpRequest, runtime_id: UUID, user_id: str, payload: UpdateAssistantUserIn
) -> Any:
    try:
        entry = await AssistantService.update_assistant_user(
            str(runtime_id),
            user_id,
            payload.can_manage,
            user=request.user,
        )
        return AssistantUserOut(
            user_id=str(entry.user_id),
            can_manage=entry.can_manage,
            created_at=entry.created_at.isoformat() if entry.created_at else "",
        )
    except PermissionDenied as e:
        return 403, Error(message=str(e))
    except ValueError as e:
        return 404, Error(message=str(e))


@router.delete(
    "/assistants/runtimes/{runtime_id}/users/{user_id}/",
    response={200: Success, 403: Error, 404: Error},
    operation_id="delete_assistant_user",
)
async def delete_assistant_user(request: HttpRequest, runtime_id: UUID, user_id: str) -> Any:
    try:
        await AssistantService.remove_assistant_user(str(runtime_id), user_id, user=request.user)
        return Success(success=True, message="User removed from assistant")
    except PermissionDenied as e:
        return 403, Error(message=str(e))
    except ValueError as e:
        return 404, Error(message=str(e))


# ── Assistant Groups ──────────────────────────────────────────────────────────


class AssistantGroupOut(Schema):
    group_id: int
    group_name: str
    created_at: str


class AddAssistantGroupIn(Schema):
    group_id: int


@router.get(
    "/assistants/runtimes/{runtime_id}/groups/",
    response={200: list[AssistantGroupOut], 403: Error, 404: Error},
    operation_id="list_assistant_groups",
)
async def list_assistant_groups(request: HttpRequest, runtime_id: UUID) -> Any:
    try:
        groups = await AssistantService.list_assistant_groups(str(runtime_id), user=request.user)
        return [
            AssistantGroupOut(
                group_id=g.group_id,
                group_name=g.group.name,
                created_at=g.created_at.isoformat() if g.created_at else "",
            )
            for g in groups
        ]
    except ValueError as e:
        return 404, Error(message=str(e))


@router.post(
    "/assistants/runtimes/{runtime_id}/groups/",
    response={200: AssistantGroupOut, 403: Error, 404: Error},
    operation_id="add_assistant_group",
)
async def add_assistant_group(
    request: HttpRequest, runtime_id: UUID, payload: AddAssistantGroupIn
) -> Any:
    try:
        entry = await AssistantService.add_assistant_group(
            str(runtime_id),
            payload.group_id,
            user=request.user,
        )
        return AssistantGroupOut(
            group_id=entry.group_id,
            group_name=entry.group.name,
            created_at=entry.created_at.isoformat() if entry.created_at else "",
        )
    except PermissionDenied as e:
        return 403, Error(message=str(e))
    except ValueError as e:
        return 404, Error(message=str(e))


@router.delete(
    "/assistants/runtimes/{runtime_id}/groups/{group_id}/",
    response={200: Success, 403: Error, 404: Error},
    operation_id="delete_assistant_group",
)
async def delete_assistant_group(request: HttpRequest, runtime_id: UUID, group_id: int) -> Any:
    try:
        await AssistantService.remove_assistant_group(str(runtime_id), group_id, user=request.user)
        return Success(success=True, message="Group removed from assistant")
    except PermissionDenied as e:
        return 403, Error(message=str(e))
    except ValueError as e:
        return 404, Error(message=str(e))
