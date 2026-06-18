from datetime import datetime
from typing import Any
from uuid import UUID

from django.http import HttpRequest
from django_ai_sdk import Assistant
from django_ai_sdk.assistants.services import AssistantService, AssistantSettingsService
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
from django_ai_sdk.workflows import WorkflowDefinition, WorkflowService
from django_ai_sdk.workflows.models import WorkflowSettings
from ninja import Router, Schema
from pydantic import ConfigDict

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


class Tool(Schema):
    label: str
    description: str | None = None
    children: list["Tool"] = []


class MCPServerStatus(Schema):
    server_name: str
    label: str
    type: str
    status: str
    tool_names: list[str]
    connect_url: str | None = None


class ToolsResponse(Schema):
    tools: list[Tool]
    mcp: list[MCPServerStatus] = []


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

    mcp_data = []
    try:
        mcp_status = await AssistantService.get_mcp_server_status(assistant, user=request.user)
        mcp_data = [
            MCPServerStatus(
                server_name=s.server_name,
                label=s.label,
                type=s.type,
                status=s.status,
                tool_names=s.tool_names,
            )
            for s in mcp_status
        ]
    except Exception:
        logger.exception("Failed to load MCP status for assistant %s", assistant_id)

    return ToolsResponse(tools=tools_data, mcp=mcp_data)


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
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    slug: str
    assistant: str
    model: str
    system_prompt: str
    tools: list[str]
    mcp_servers: list[str]
    suggestion_enabled: bool
    title_generation: bool
    max_history: int | None
    file_upload: bool
    active: bool
    created_at: datetime
    updated_at: datetime


class AssistantSettingsCreateIn(Schema):
    name: str
    slug: str = ""
    assistant: str = ""
    model: str = "gpt-4o"
    system_prompt: str = ""
    tools: list[str] = []
    mcp_servers: list[str] = []
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
    mcp_servers: list[str] | None = None
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
    return await AssistantSettingsService.all()


@router.post(
    "/assistants/runtimes/",
    response={200: AssistantSettingsOut, 400: Error},
    operation_id="create_runtime_assistant",
)
async def create_runtime_assistant(request: HttpRequest, payload: AssistantSettingsCreateIn) -> Any:
    try:
        return await AssistantSettingsService.create(
            {
                "name": payload.name,
                "slug": payload.slug,
                "assistant": payload.assistant,
                "model": payload.model,
                "system_prompt": payload.system_prompt,
                "tools": payload.tools,
                "mcp_servers": payload.mcp_servers,
                "suggestion_enabled": payload.suggestion_enabled,
                "title_generation": payload.title_generation,
                "max_history": payload.max_history,
                "file_upload": payload.file_upload,
            },
            user=request.user,
        )
    except Exception as e:
        return 400, Error(message=str(e))


@router.get(
    "/assistants/runtimes/{runtime_id}/",
    response={200: AssistantSettingsOut, 404: Error},
    operation_id="get_runtime_assistant",
)
async def get_runtime_assistant(request: HttpRequest, runtime_id: UUID) -> Any:
    try:
        return await AssistantSettingsService.get(str(runtime_id))
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
        return await AssistantSettingsService.update(str(runtime_id), data)
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
        return await AssistantSettingsService.delete(str(runtime_id))
    except ValueError as e:
        return 404, Error(message=str(e))


# ============================================================================
# Workflows
# ============================================================================


class WorkflowRunRequest(Schema):
    workflow: WorkflowDefinition
    messages: list[dict] = []


class WorkflowRunResponse(Schema):
    outputs: dict


class WorkflowActionItem(Schema):
    key: str
    description: str


@router.post(
    "/workflows/run/",
    response={200: WorkflowRunResponse, 400: Error, 404: Error, 500: Error},
    operation_id="run_workflow",
)
async def run_workflow(request: HttpRequest, payload: WorkflowRunRequest) -> Any:
    try:
        outputs = await WorkflowService.run(payload.workflow, payload.messages, user=request.user)
        return WorkflowRunResponse(outputs=outputs)
    except ValueError as e:
        return 404, Error(message=str(e))
    except Exception as e:
        return 500, Error(message=str(e))


@router.get(
    "/workflows/actions/",
    response={200: list[WorkflowActionItem]},
    operation_id="list_workflow_actions",
)
def list_workflow_actions(request: HttpRequest) -> list[WorkflowActionItem]:
    return [WorkflowActionItem(**item) for item in WorkflowService.list_actions()]


# Workflow CRUD schemas


class WorkflowCreateRequest(Schema):
    name: str
    workflow: WorkflowDefinition


class WorkflowUpdateRequest(Schema):
    name: str | None = None
    workflow: WorkflowDefinition | None = None
    active: bool | None = None


class WorkflowItem(Schema):
    id: str
    name: str
    definition: dict
    active: bool


class WorkflowRunByIdRequest(Schema):
    messages: list[dict] = []


@router.get(
    "/workflows/",
    response={200: list[WorkflowItem]},
    operation_id="list_workflows",
)
async def list_workflows(request: HttpRequest) -> Any:
    records = await WorkflowService.list_workflows()
    return [
        WorkflowItem(id=str(r.id), name=r.name, definition=r.definition, active=r.active)
        for r in records
    ]


@router.post(
    "/workflows/",
    response={201: WorkflowItem, 400: Error, 500: Error},
    operation_id="create_workflow",
)
async def create_workflow(request: HttpRequest, payload: WorkflowCreateRequest) -> Any:
    try:
        record = await WorkflowService.create(payload.name, payload.workflow, user=request.user)
        return 201, WorkflowItem(
            id=str(record.id), name=record.name, definition=record.definition, active=record.active
        )
    except Exception as e:
        return 500, Error(message=str(e))


@router.get(
    "/workflows/{workflow_id}/",
    response={200: WorkflowItem, 404: Error},
    operation_id="get_workflow",
)
async def get_workflow(request: HttpRequest, workflow_id: str) -> Any:
    try:
        record = await WorkflowService.get(workflow_id)
        return WorkflowItem(
            id=str(record.id), name=record.name, definition=record.definition, active=record.active
        )
    except WorkflowSettings.DoesNotExist:
        return 404, Error(message="Workflow not found")


@router.patch(
    "/workflows/{workflow_id}/",
    response={200: WorkflowItem, 404: Error, 500: Error},
    operation_id="update_workflow",
)
async def update_workflow(
    request: HttpRequest, workflow_id: str, payload: WorkflowUpdateRequest
) -> Any:
    try:
        record = await WorkflowService.update(
            workflow_id,
            name=payload.name,
            workflow=payload.workflow,
            active=payload.active,
        )
        return WorkflowItem(
            id=str(record.id), name=record.name, definition=record.definition, active=record.active
        )
    except WorkflowSettings.DoesNotExist:
        return 404, Error(message="Workflow not found")
    except Exception as e:
        return 500, Error(message=str(e))


@router.delete(
    "/workflows/{workflow_id}/",
    response={204: None, 404: Error},
    operation_id="delete_workflow",
)
async def delete_workflow(request: HttpRequest, workflow_id: str) -> Any:
    try:
        await WorkflowService.get(workflow_id)  # raises DoesNotExist if missing
        await WorkflowService.delete(workflow_id)
        return 204, None
    except WorkflowSettings.DoesNotExist:
        return 404, Error(message="Workflow not found")


@router.post(
    "/workflows/{workflow_id}/run/",
    response={200: WorkflowRunResponse, 404: Error, 500: Error},
    operation_id="run_workflow_by_id",
)
async def run_workflow_by_id(
    request: HttpRequest, workflow_id: str, payload: WorkflowRunByIdRequest
) -> Any:
    try:
        outputs = await WorkflowService.run_by_id(workflow_id, payload.messages, user=request.user)
        return WorkflowRunResponse(outputs=outputs)
    except WorkflowSettings.DoesNotExist:
        return 404, Error(message="Workflow not found")
    except Exception as e:
        return 500, Error(message=str(e))
