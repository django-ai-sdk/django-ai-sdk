from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from django.contrib.auth import get_user_model
from django.db.models import Q
from django.http import HttpRequest
from django_ai_sdk import Agent
from django_ai_sdk.agents.services import AgentService
from django_ai_sdk.common import ChatMessage
from django_ai_sdk.logger import get_logger
from django_ai_sdk.memories.services import MemoryService
from django_ai_sdk.permissions import ObjectPermissions, Operation, PermissionDenied
from django_ai_sdk.storage.schemas import (
    ThreadInfo,  # noqa: TC002 — needed at runtime for Pydantic schema
)
from django_ai_sdk.storage.services import (
    ThreadService,
    aget_thread_file_meta,
    aget_thread_history,
)
from django_ai_sdk.views.schemas import ChatRequest, RateMessagePayload
from django_ai_sdk.workflows import WorkflowDefinition, WorkflowService
from django_ai_sdk.workflows.models import WorkflowSettings
from ninja import Router, Schema

from .permissions import agent_permissions, thread_permissions

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


class AgentItem(Schema):
    id: str
    name: str | None = None
    model: str | None = None
    file_upload: bool = False
    rag: bool = False


class AgentsListResponse(Schema):
    agents: list[AgentItem]


class AgentInfoResponse(Schema):
    id: str
    name: str | None = None
    model: str | None = None
    class_name: str
    description: str | None = None
    instructions: str | None = None
    file_upload: bool = False
    rag: bool = False
    permissions: ObjectPermissions = ObjectPermissions()


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
    agent_id: str
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
    permissions: ObjectPermissions = ObjectPermissions()


class ThreadFileMeta(Schema):
    file_count: int = 0
    file_memory_id: str | None = None


class DeleteAllThreadsResponse(Schema):
    success: bool
    deleted_count: int


class PatchThreadPayload(Schema):
    agent_id: str


class MessageResponse(Schema):
    id: str
    is_deleted: bool = False
    feedback: FeedbackResponse | None = None


class RunResponse(Schema):
    result: str | None = None
    thread_id: str


@router.get("/health/", response={200: HealthResponse}, operation_id="health_check")
def health_check(request: HttpRequest) -> HealthResponse:
    return HealthResponse(status="ok", service="django-ai-sdk")


@router.get(
    "/threads/", response={200: ThreadListResponse, 500: Error}, operation_id="list_threads"
)
async def list_threads(request: HttpRequest, limit: int = 100, offset: int = 0) -> Any:
    try:
        all_threads = await ThreadService.threads(user=request.user, limit=limit, offset=offset)
        items = [
            ThreadListItem(
                id=t.id,
                title=t.title,
                agent_id=t.agent_id,
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
        agent_id = payload.agent_id or ""
        # Initial messages are not persisted here; the chat/stream endpoint
        # receives and stores the full message list.
        thread_id = await ThreadService.create_thread(
            agent_id=agent_id,
            user=request.user,
        )
        await MemoryService.link_memories(agent_id, thread_id, user=request.user)
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

        perms = await thread_permissions(request.user, thread_id)
        return ThreadDetailResponse(**data, permissions=perms)
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
    "/agents/{agent_id}/run/",
    response={200: RunResponse, 400: Error, 403: Error, 404: Error, 500: Error, 501: Error},
    operation_id="run_agent",
)
async def run_agent(request: HttpRequest, agent_id: str, payload: ChatRequest) -> Any:
    try:
        agent = await AgentService.get(agent_id)
        chat_messages = agent.protocol_handler.to_chat_messages(payload.messages)
        result = await agent.run(chat_messages, user=request.user)
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
        agent = await AgentService.get_agent(thread_id, user=request.user)
        chat_messages = agent.protocol_handler.to_chat_messages(payload.messages)
        result = await agent.run(chat_messages, thread_id=thread_id, user=request.user)
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
        agent = await AgentService.get_agent(thread_id, user=request.user)
        return await agent.as_view(payload.messages, thread_id=thread_id, user=request.user)
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
        await AgentService.get(payload.agent_id)
        thread = await ThreadService.get_thread(thread_id, user=request.user)
        if thread is None:
            return 404, Error(message="Thread not found")
        if thread.agent_id:
            await MemoryService.unlink_memories(thread.agent_id, thread_id, user=request.user)
        await ThreadService.update_thread(
            thread_id, metadata={"agent_id": payload.agent_id}, user=request.user
        )
        await MemoryService.link_memories(payload.agent_id, thread_id, user=request.user)
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


# ============================================================================
# Runtime Agents (DB-configured)
# ============================================================================


class AgentSettingsOut(Schema):
    id: UUID
    name: str
    slug: str
    agent: str
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


class AgentSettingsCreateUserEntry(Schema):
    user_id: str
    can_manage: bool = False


class AgentSettingsCreateGroupEntry(Schema):
    group_id: int
    can_manage: bool = False


class AgentSettingsCreateIn(Schema):
    name: str
    slug: str = ""
    agent: str = ""
    model: str = "gpt-4o"
    system_prompt: str = ""
    tools: list[str] = []
    integrations: list[str] = []
    memories: list[str] = []
    users: list[AgentSettingsCreateUserEntry] = []
    groups: list[AgentSettingsCreateGroupEntry] = []
    suggestion_enabled: bool = False
    title_generation: bool = True
    max_history: int | None = None
    file_upload: bool = False


class AgentSettingsUpdateIn(Schema):
    name: str | None = None
    agent: str | None = None
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


class RuntimeAgentBaseItem(Schema):
    path: str
    name: str


class RuntimeAgentToolItem(Schema):
    key: str
    path: str


@router.get(
    "/agents/runtimes/bases/",
    response={200: list[RuntimeAgentBaseItem]},
    operation_id="list_runtime_agent_bases",
)
def list_runtime_agent_bases(request: HttpRequest) -> list[RuntimeAgentBaseItem]:
    from django_ai_sdk.agents.config import get_runtime_agent_bases

    return [
        RuntimeAgentBaseItem(
            path=f"{cls.__module__}.{cls.__qualname__}",
            name=cls.__name__,
        )
        for cls in get_runtime_agent_bases()
    ]


@router.get(
    "/agents/runtimes/tools/",
    response={200: list[RuntimeAgentToolItem]},
    operation_id="list_runtime_agent_tools",
)
def list_runtime_agent_tools(request: HttpRequest) -> list[RuntimeAgentToolItem]:
    from django_ai_sdk.agents.config import get_tool_registry

    return [RuntimeAgentToolItem(key=key, path=path) for key, path in get_tool_registry().items()]


@router.get(
    "/agents/runtimes/",
    response={200: list[AgentSettingsOut]},
    operation_id="list_runtime_agents",
)
async def list_runtime_agents(request: HttpRequest) -> Any:
    return await AgentService.list_runtime_agents(user=request.user)


@router.post(
    "/agents/runtimes/",
    response={200: AgentSettingsOut, 400: Error},
    operation_id="create_runtime_agent",
)
async def create_runtime_agent(request: HttpRequest, payload: AgentSettingsCreateIn) -> Any:
    try:
        config = await AgentService.create_runtime_agent(
            {
                "name": payload.name,
                "slug": payload.slug,
                "agent": payload.agent,
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
                await AgentService.add_agent_user(
                    str(config.id), entry.user_id, entry.can_manage, user=request.user
                )
            except Exception:
                pass
        for entry in payload.groups:
            try:
                await AgentService.add_agent_group(
                    str(config.id), entry.group_id, entry.can_manage, user=request.user
                )
            except Exception:
                pass
        return config
    except Exception as e:
        return 400, Error(message=str(e))


@router.get(
    "/agents/runtimes/{runtime_id}/",
    response={200: AgentSettingsOut, 404: Error},
    operation_id="get_runtime_agent",
)
async def get_runtime_agent(request: HttpRequest, runtime_id: UUID) -> Any:
    try:
        return await AgentService.get_runtime_agent(str(runtime_id), user=request.user)
    except ValueError as e:
        return 404, Error(message=str(e))


@router.patch(
    "/agents/runtimes/{runtime_id}/",
    response={200: AgentSettingsOut, 404: Error, 400: Error},
    operation_id="update_runtime_agent",
)
async def update_runtime_agent(
    request: HttpRequest, runtime_id: UUID, payload: AgentSettingsUpdateIn
) -> Any:
    try:
        from typing import cast

        from django_ai_sdk.agents.services import AgentUpdateData

        data = cast(
            "AgentUpdateData",
            {k: v for k, v in payload.model_dump().items() if v is not None},
        )
        return await AgentService.update_runtime_agent(str(runtime_id), data, user=request.user)
    except ValueError as e:
        return 404, Error(message=str(e))
    except Exception as e:
        return 400, Error(message=str(e))


@router.delete(
    "/agents/runtimes/{runtime_id}/",
    response={200: AgentSettingsOut, 404: Error},
    operation_id="delete_runtime_agent",
)
async def delete_runtime_agent(request: HttpRequest, runtime_id: UUID) -> Any:
    try:
        return await AgentService.delete_runtime_agent(str(runtime_id), user=request.user)
    except ValueError as e:
        return 404, Error(message=str(e))


# ── Agent Users ───────────────────────────────────────────────────────────


class AgentUserOut(Schema):
    user_id: str
    email: str = ""
    first_name: str = ""
    last_name: str = ""
    can_manage: bool
    created_at: str


class AddAgentUserIn(Schema):
    user_id: str
    can_manage: bool = False


class UpdateAgentUserIn(Schema):
    can_manage: bool


@router.get(
    "/agents/runtimes/{runtime_id}/users/",
    response={200: list[AgentUserOut], 403: Error, 404: Error},
    operation_id="list_agent_users",
)
async def list_agent_users(request: HttpRequest, runtime_id: UUID) -> Any:
    try:
        users = await AgentService.list_agent_users(str(runtime_id), user=request.user)
        return [
            AgentUserOut(
                user_id=str(u.user_id),
                email=u.user.email,
                first_name=u.user.first_name,
                last_name=u.user.last_name,
                can_manage=u.can_manage,
                created_at=u.created_at.isoformat() if u.created_at else "",
            )
            for u in users
        ]
    except ValueError as e:
        return 404, Error(message=str(e))


@router.post(
    "/agents/runtimes/{runtime_id}/users/",
    response={200: AgentUserOut, 403: Error, 404: Error},
    operation_id="add_agent_user",
)
async def add_agent_user(request: HttpRequest, runtime_id: UUID, payload: AddAgentUserIn) -> Any:
    try:
        entry = await AgentService.add_agent_user(
            str(runtime_id),
            payload.user_id,
            payload.can_manage,
            user=request.user,
        )
        return AgentUserOut(
            user_id=str(entry.user_id),
            email=entry.user.email,
            first_name=entry.user.first_name,
            last_name=entry.user.last_name,
            can_manage=entry.can_manage,
            created_at=entry.created_at.isoformat() if entry.created_at else "",
        )
    except PermissionDenied as e:
        return 403, Error(message=str(e))
    except ValueError as e:
        return 404, Error(message=str(e))


@router.patch(
    "/agents/runtimes/{runtime_id}/users/{user_id}/",
    response={200: AgentUserOut, 403: Error, 404: Error},
    operation_id="update_agent_user",
)
async def update_agent_user(
    request: HttpRequest, runtime_id: UUID, user_id: str, payload: UpdateAgentUserIn
) -> Any:
    try:
        entry = await AgentService.update_agent_user(
            str(runtime_id),
            user_id,
            payload.can_manage,
            user=request.user,
        )
        return AgentUserOut(
            user_id=str(entry.user_id),
            email=entry.user.email,
            first_name=entry.user.first_name,
            last_name=entry.user.last_name,
            can_manage=entry.can_manage,
            created_at=entry.created_at.isoformat() if entry.created_at else "",
        )
    except PermissionDenied as e:
        return 403, Error(message=str(e))
    except ValueError as e:
        return 404, Error(message=str(e))


@router.delete(
    "/agents/runtimes/{runtime_id}/users/{user_id}/",
    response={200: Success, 403: Error, 404: Error},
    operation_id="delete_agent_user",
)
async def delete_agent_user(request: HttpRequest, runtime_id: UUID, user_id: str) -> Any:
    try:
        await AgentService.remove_agent_user(str(runtime_id), user_id, user=request.user)
        return Success(success=True, message="User removed from agent")
    except PermissionDenied as e:
        return 403, Error(message=str(e))
    except ValueError as e:
        return 404, Error(message=str(e))


# ── Agent Groups ──────────────────────────────────────────────────────────


class AgentGroupOut(Schema):
    group_id: int
    group_name: str
    can_manage: bool
    created_at: str


class AddAgentGroupIn(Schema):
    group_id: int
    can_manage: bool = False


@router.get(
    "/agents/runtimes/{runtime_id}/groups/",
    response={200: list[AgentGroupOut], 403: Error, 404: Error},
    operation_id="list_agent_groups",
)
async def list_agent_groups(request: HttpRequest, runtime_id: UUID) -> Any:
    try:
        groups = await AgentService.list_agent_groups(str(runtime_id), user=request.user)
        return [
            AgentGroupOut(
                group_id=g.group_id,
                group_name=g.group.name,
                can_manage=g.can_manage,
                created_at=g.created_at.isoformat() if g.created_at else "",
            )
            for g in groups
        ]
    except ValueError as e:
        return 404, Error(message=str(e))


@router.post(
    "/agents/runtimes/{runtime_id}/groups/",
    response={200: AgentGroupOut, 403: Error, 404: Error},
    operation_id="add_agent_group",
)
async def add_agent_group(request: HttpRequest, runtime_id: UUID, payload: AddAgentGroupIn) -> Any:
    try:
        entry = await AgentService.add_agent_group(
            str(runtime_id),
            payload.group_id,
            payload.can_manage,
            user=request.user,
        )
        return AgentGroupOut(
            group_id=entry.group_id,
            group_name=entry.group.name,
            can_manage=entry.can_manage,
            created_at=entry.created_at.isoformat() if entry.created_at else "",
        )
    except PermissionDenied as e:
        return 403, Error(message=str(e))
    except ValueError as e:
        return 404, Error(message=str(e))


@router.delete(
    "/agents/runtimes/{runtime_id}/groups/{group_id}/",
    response={200: Success, 403: Error, 404: Error},
    operation_id="delete_agent_group",
)
async def delete_agent_group(request: HttpRequest, runtime_id: UUID, group_id: int) -> Any:
    try:
        await AgentService.remove_agent_group(str(runtime_id), group_id, user=request.user)
        return Success(success=True, message="Group removed from agent")
    except PermissionDenied as e:
        return 403, Error(message=str(e))
    except ValueError as e:
        return 404, Error(message=str(e))


@router.get(
    "/agents/",
    response={200: AgentsListResponse, 403: Error, 500: Error},
    operation_id="list_agents",
)
async def list_agents(request: HttpRequest, limit: int = 100, offset: int = 0) -> Any:
    try:
        items = await AgentService.list_agents(user=request.user, limit=limit, offset=offset)
        return AgentsListResponse(agents=[AgentItem(**item) for item in items])
    except PermissionDenied as e:
        return 403, Error(message=str(e))
    except Exception as e:
        return 500, Error(message=str(e))


@router.get(
    "/agents/{agent_id}/",
    response={200: AgentInfoResponse, 403: Error, 404: Error},
    operation_id="get_agent_info",
)
async def get_agent_info(request: HttpRequest, agent_id: str) -> Any:
    try:
        agent = await AgentService.get(agent_id)
        info = await AgentService.get_agent_info(agent_id, user=request.user)
        perms = await agent_permissions(request.user, agent_id)
        return AgentInfoResponse(
            id=info.id,
            name=info.name,
            model=info.model,
            class_name=info.class_name,
            description=info.description,
            instructions=agent.get_system_prompt(),
            file_upload=info.file_upload,
            rag=info.rag,
            permissions=perms,
        )
    except PermissionDenied as e:
        return 403, Error(message=str(e))
    except ValueError as e:
        return 404, Error(message=str(e))


@router.get(
    "/agents/{agent_id}/tools/",
    response={200: ToolsResponse, 403: Error, 404: Error},
    operation_id="get_agent_tools",
)
async def get_agent_tools(request: HttpRequest, agent_id: str) -> Any:
    try:
        agent = await AgentService.get(agent_id)
        await AgentService.has_perms(
            request.user,
            Operation.VIEW_AGENT,
            obj=getattr(agent, "_config", None),
            agent=agent,
        )
    except ValueError as e:
        return 404, Error(message=str(e))
    except PermissionDenied as e:
        return 403, Error(message=str(e))

    tools_data = []
    try:
        tool_objs = await agent.get_tools()
        tools_data = [
            Tool(
                label=getattr(t, "label", None) or t.name.replace("_", " ").title(),
                description=t.description or "",
            )
            for t in tool_objs
        ]
    except Exception:
        logger.exception("Failed to build tools for agent %s", agent_id)

    integrations_data = []
    try:
        integration_status = await AgentService.get_integration_status(agent, user=request.user)
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
        logger.exception("Failed to load integration status for agent %s", agent_id)

    return ToolsResponse(tools=tools_data, integrations=integrations_data)


@router.post(
    "/agents/{agent_id}/reindex/",
    response={200: Success, 404: Error, 500: Error},
    operation_id="reindex_agent",
)
async def reindex_agent(
    request: HttpRequest,
    agent_id: str,
    memory_id: str | None = None,
    force_rebuild: bool = False,
) -> Any:
    try:
        agent = await AgentService.get(agent_id)
        result = await Agent.reindex(agent, memory_id, force_rebuild)

        if not result:
            return Success(success=False, message="No RAG provider configured for this agent")

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
# Workflows
# ============================================================================


class WorkflowRunRequest(Schema):
    workflow: WorkflowDefinition
    messages: list[ChatMessage] = []


class WorkflowRunResponse(Schema):
    run_id: str
    status: str


class WorkflowActionItem(Schema):
    key: str
    description: str


class WorkflowRunStepOut(Schema):
    id: str
    sequence: int
    step_name: str
    output_key: str
    output: dict | None = None
    status: str
    error: str
    started_at: str | None = None
    completed_at: str | None = None


class WorkflowRunOut(Schema):
    id: str
    workflow_id: str | None = None
    status: str
    outputs: dict | None = None
    error: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None


class WorkflowRunDetailOut(WorkflowRunOut):
    steps: list[WorkflowRunStepOut] = []


@router.post(
    "/workflows/run/",
    response={202: WorkflowRunResponse, 400: Error, 500: Error},
    operation_id="run_workflow",
)
async def run_workflow(request: HttpRequest, payload: WorkflowRunRequest) -> Any:
    try:
        run = await WorkflowService.run(payload.workflow, payload.messages, user=request.user)
        return 202, WorkflowRunResponse(run_id=str(run.id), status=run.status)
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
    messages: list[ChatMessage] = []
    run_id: str | None = None


@router.get(
    "/workflows/",
    response={200: list[WorkflowItem]},
    operation_id="list_workflows",
)
async def list_workflows(request: HttpRequest, limit: int = 100, offset: int = 0) -> Any:
    records = await WorkflowService.list_workflows(limit=limit, offset=offset)
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
    "/workflows/{workflow_id}/runs/",
    response={200: list[WorkflowRunOut], 500: Error},
    operation_id="list_workflow_runs",
)
async def list_workflow_runs(
    request: HttpRequest, workflow_id: str, limit: int = 50, offset: int = 0
) -> Any:
    try:
        runs = await WorkflowService.list_runs(workflow_id, limit=limit, offset=offset)
        return [
            WorkflowRunOut(
                id=str(r.id),
                workflow_id=str(r.workflow_id) if r.workflow_id else None,
                status=r.status,
                outputs=r.outputs,
                error=r.error,
                created_at=r.created_at.isoformat(),
                started_at=r.started_at.isoformat() if r.started_at else None,
                completed_at=r.completed_at.isoformat() if r.completed_at else None,
            )
            for r in runs
        ]
    except Exception as e:
        return 500, Error(message=str(e))


@router.get(
    "/workflows/{workflow_id}/runs/{run_id}/",
    response={200: WorkflowRunDetailOut, 404: Error, 500: Error},
    operation_id="get_workflow_run",
)
async def get_workflow_run(request: HttpRequest, workflow_id: str, run_id: str) -> Any:
    from django_ai_sdk.workflows.models import WorkflowRun

    try:
        run = await WorkflowService.get_run(run_id)
        steps = [
            WorkflowRunStepOut(
                id=str(s.id),
                sequence=s.sequence,
                step_name=s.step_name,
                output_key=s.output_key,
                output=s.output if isinstance(s.output, dict) else None,
                status=s.status,
                error=s.error,
                started_at=s.started_at.isoformat() if s.started_at else None,
                completed_at=s.completed_at.isoformat() if s.completed_at else None,
            )
            async for s in run.steps.all()
        ]
        return WorkflowRunDetailOut(
            id=str(run.id),
            workflow_id=str(run.workflow_id) if run.workflow_id else None,
            status=run.status,
            outputs=run.outputs,
            error=run.error,
            created_at=run.created_at.isoformat(),
            started_at=run.started_at.isoformat() if run.started_at else None,
            completed_at=run.completed_at.isoformat() if run.completed_at else None,
            steps=steps,
        )
    except WorkflowRun.DoesNotExist:
        return 404, Error(message="Run not found")
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
        await WorkflowService.get(workflow_id)
        await WorkflowService.delete(workflow_id)
        return 204, None
    except WorkflowSettings.DoesNotExist:
        return 404, Error(message="Workflow not found")


@router.post(
    "/workflows/{workflow_id}/run/",
    response={202: WorkflowRunResponse, 404: Error, 500: Error},
    operation_id="run_workflow_by_id",
)
async def run_workflow_by_id(
    request: HttpRequest, workflow_id: str, payload: WorkflowRunByIdRequest
) -> Any:
    try:
        run = await WorkflowService.run_by_id(
            workflow_id, payload.messages, user=request.user, run_id=payload.run_id
        )
        return 202, WorkflowRunResponse(run_id=str(run.id), status=run.status)
    except WorkflowSettings.DoesNotExist:
        return 404, Error(message="Workflow not found")
    except Exception as e:
        return 500, Error(message=str(e))


class UserSchema(Schema):
    id: Any
    first_name: str
    last_name: str


class UserDetailSchema(Schema):
    id: Any
    first_name: str
    last_name: str
    email: str


class UserUpdateSchema(Schema):
    first_name: str | None = None
    last_name: str | None = None


@router.get("/users/", response=list[UserSchema], operation_id="list_users")
def list_users(request: HttpRequest, q: str = "", limit: int = 10) -> Any:
    User = get_user_model()
    qs = User.objects.order_by("first_name", "last_name")
    if q.strip():
        qs = qs.filter(
            Q(first_name__icontains=q)
            | Q(last_name__icontains=q)
            | Q(email__icontains=q)
            | Q(username__icontains=q)
        )
    return list(qs.values("id", "first_name", "last_name")[: min(limit, 100)])


@router.get("/users/me/", response=UserDetailSchema, operation_id="get_me")
def get_me(request: HttpRequest) -> Any:
    User = get_user_model()
    return User.objects.get(pk=request.user.pk)


@router.patch("/users/me/", response=UserDetailSchema, operation_id="update_me")
def update_me(request: HttpRequest, payload: UserUpdateSchema) -> Any:
    User = get_user_model()
    user = User.objects.get(pk=request.user.pk)
    update_fields = []
    for field, value in payload.model_dump(exclude_none=True).items():
        setattr(user, field, value)
        update_fields.append(field)
    if update_fields:
        user.save(update_fields=update_fields)
    return user


class GroupOut(Schema):
    id: int
    name: str


@router.get("/accounts/groups/", response=list[GroupOut], operation_id="search_groups")
def search_groups(request: HttpRequest, q: str = "", limit: int = 10) -> Any:
    from django.contrib.auth.models import Group

    qs = Group.objects.order_by("name")
    if q.strip():
        qs = qs.filter(name__icontains=q)
    return list(qs.values("id", "name")[: min(limit, 100)])
