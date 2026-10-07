"""Agents: code-declared and runtime (DB-configured), their tools, users and groups."""

from __future__ import annotations

import contextlib
from collections.abc import Collection
from typing import Any, cast
from uuid import UUID

from django.http import HttpRequest
from ninja import Router

from django_ai_sdk import Agent
from django_ai_sdk.agents.services import AgentService, AgentUpdateData
from django_ai_sdk.contrib.ninja.routing import ApiRouter, Limit, Offset
from django_ai_sdk.contrib.ninja.schemas import (
    AgentGroupOut,
    AgentInfoResponse,
    AgentItem,
    AgentSettingsOut,
    AgentsListResponse,
    AgentUserOut,
    IntegrationStatusOut,
    RunResponse,
    RuntimeAgentBaseItem,
    RuntimeAgentToolItem,
    Success,
    Tool,
    ToolsResponse,
)
from django_ai_sdk.logger import get_logger
from django_ai_sdk.permissions import Operation, PermissionDenied
from django_ai_sdk.views.permissions import aagent_permissions
from django_ai_sdk.views.schemas import (
    AddAgentGroupIn,
    AddAgentUserIn,
    AgentSettingsCreateIn,
    AgentSettingsUpdateIn,
    ChatRequest,
    UpdateAgentUserIn,
)

logger = get_logger(__name__)
routes = ApiRouter()


def _user_out(entry: Any) -> AgentUserOut:
    return AgentUserOut(
        user_id=str(entry.user_id),
        email=getattr(entry.user, "email", "") or "",
        first_name=getattr(entry.user, "first_name", "") or "",
        last_name=getattr(entry.user, "last_name", "") or "",
        can_manage=entry.can_manage,
        created_at=entry.created_at.isoformat() if entry.created_at else "",
    )


def _group_out(entry: Any) -> AgentGroupOut:
    return AgentGroupOut(
        group_id=entry.group_id,
        group_name=entry.group.name,
        can_manage=entry.can_manage,
        created_at=entry.created_at.isoformat() if entry.created_at else "",
    )


# Runtime agents come first: "/agents/runtimes/" must win over "/agents/{agent_id}/".


@routes.get("/agents/runtimes/bases/", response=list[RuntimeAgentBaseItem])
def list_runtime_agent_bases(request: HttpRequest) -> Any:
    from django_ai_sdk.agents.config import get_runtime_agent_bases

    return [
        RuntimeAgentBaseItem(
            path=f"{cls.__module__}.{cls.__qualname__}",
            name=cls.__name__,
            base_system_prompt=getattr(cls, "base_system_prompt", None),
        )
        for cls in get_runtime_agent_bases()
    ]


@routes.get("/agents/runtimes/tools/", response=list[RuntimeAgentToolItem])
def list_runtime_agent_tools(request: HttpRequest) -> Any:
    from django_ai_sdk.agents.config import get_tool_registry

    return [RuntimeAgentToolItem(key=key, path=path) for key, path in get_tool_registry().items()]


async def _runtime_out(request: HttpRequest, config: Any) -> AgentSettingsOut:
    out = AgentSettingsOut.model_validate(config)
    out.permissions = await AgentService.get_object_permissions(request.user, config)
    return out


@routes.get("/agents/runtimes/", response=list[AgentSettingsOut])
async def list_runtime_agents(request: HttpRequest) -> Any:
    configs = await AgentService.list_runtime_agents(user=request.user)
    return [await _runtime_out(request, config) for config in configs]


@routes.post("/agents/runtimes/", response=AgentSettingsOut)
async def create_runtime_agent(request: HttpRequest, payload: AgentSettingsCreateIn) -> Any:
    config = await AgentService.create_runtime_agent(
        payload.model_dump(exclude={"users", "groups"}),  # type: ignore[arg-type]
        user=request.user,
    )
    # Sharing is best effort: the agent exists either way, and the owner can retry.
    for entry in payload.users:
        with contextlib.suppress(PermissionDenied, ValueError):
            await AgentService.add_agent_user(
                str(config.id), entry.user_id, entry.can_manage, user=request.user
            )
    for group in payload.groups:
        with contextlib.suppress(PermissionDenied, ValueError):
            await AgentService.add_agent_group(
                str(config.id), group.group_id, group.can_manage, user=request.user
            )
    return await _runtime_out(request, config)


@routes.get("/agents/runtimes/{runtime_id}/", response=AgentSettingsOut)
async def get_runtime_agent(request: HttpRequest, runtime_id: UUID) -> Any:
    config = await AgentService.get_runtime_agent(str(runtime_id), user=request.user)
    return await _runtime_out(request, config)


@routes.patch("/agents/runtimes/{runtime_id}/", response=AgentSettingsOut)
async def update_runtime_agent(
    request: HttpRequest, runtime_id: UUID, payload: AgentSettingsUpdateIn
) -> Any:
    data = cast("AgentUpdateData", payload.model_dump(exclude_unset=True))
    config = await AgentService.update_runtime_agent(str(runtime_id), data, user=request.user)
    return await _runtime_out(request, config)


@routes.delete("/agents/runtimes/{runtime_id}/", response=Success)
async def delete_runtime_agent(request: HttpRequest, runtime_id: UUID) -> Any:
    await AgentService.delete_runtime_agent(str(runtime_id), user=request.user)
    return Success(success=True, message="Agent deleted successfully")


@routes.get("/agents/runtimes/{runtime_id}/users/", response=list[AgentUserOut])
async def list_agent_users(request: HttpRequest, runtime_id: UUID) -> Any:
    users = await AgentService.list_agent_users(str(runtime_id), user=request.user)
    return [_user_out(u) for u in users]


@routes.post("/agents/runtimes/{runtime_id}/users/", response=AgentUserOut)
async def add_agent_user(request: HttpRequest, runtime_id: UUID, payload: AddAgentUserIn) -> Any:
    entry = await AgentService.add_agent_user(
        str(runtime_id), payload.user_id, payload.can_manage, user=request.user
    )
    return _user_out(entry)


@routes.patch("/agents/runtimes/{runtime_id}/users/{user_id}/", response=AgentUserOut)
async def update_agent_user(
    request: HttpRequest, runtime_id: UUID, user_id: str, payload: UpdateAgentUserIn
) -> Any:
    entry = await AgentService.update_agent_user(
        str(runtime_id), user_id, payload.can_manage, user=request.user
    )
    return _user_out(entry)


@routes.delete("/agents/runtimes/{runtime_id}/users/{user_id}/", response=Success)
async def delete_agent_user(request: HttpRequest, runtime_id: UUID, user_id: str) -> Any:
    await AgentService.remove_agent_user(str(runtime_id), user_id, user=request.user)
    return Success(success=True, message="User removed from agent")


@routes.get("/agents/runtimes/{runtime_id}/groups/", response=list[AgentGroupOut])
async def list_agent_groups(request: HttpRequest, runtime_id: UUID) -> Any:
    groups = await AgentService.list_agent_groups(str(runtime_id), user=request.user)
    return [_group_out(g) for g in groups]


@routes.post("/agents/runtimes/{runtime_id}/groups/", response=AgentGroupOut)
async def add_agent_group(request: HttpRequest, runtime_id: UUID, payload: AddAgentGroupIn) -> Any:
    entry = await AgentService.add_agent_group(
        str(runtime_id), payload.group_id, payload.can_manage, user=request.user
    )
    return _group_out(entry)


@routes.delete("/agents/runtimes/{runtime_id}/groups/{group_id}/", response=Success)
async def delete_agent_group(request: HttpRequest, runtime_id: UUID, group_id: int) -> Any:
    await AgentService.remove_agent_group(str(runtime_id), group_id, user=request.user)
    return Success(success=True, message="Group removed from agent")


@routes.get("/agents/", response=AgentsListResponse)
async def list_agents(
    request: HttpRequest,
    limit: Limit = 100,
    offset: Offset = 0,
) -> Any:
    items = await AgentService.list_agents(user=request.user, limit=limit, offset=offset)
    return AgentsListResponse(agents=[AgentItem(**item) for item in items])


@routes.get("/agents/{agent_id}/", response=AgentInfoResponse)
async def get_agent_info(request: HttpRequest, agent_id: str) -> Any:
    agent = await AgentService.get(agent_id)
    info = await AgentService.get_agent_info(agent_id, user=request.user)
    return AgentInfoResponse(
        id=info.id,
        name=info.name,
        model=info.model,
        class_name=info.class_name,
        description=info.description,
        instructions=agent.get_system_prompt(),
        file_upload=info.file_upload,
        rag=info.rag,
        permissions=await aagent_permissions(request.user, agent_id),
    )


@routes.get("/agents/{agent_id}/tools/", response=ToolsResponse)
async def get_agent_tools(request: HttpRequest, agent_id: str) -> Any:
    agent = await AgentService.get(agent_id)
    await AgentService.has_perms(
        request.user,
        Operation.VIEW_AGENT,
        obj=agent.config if agent.is_runtime else None,
        agent=agent,
    )
    # A broken tool or integration must not hide the rest: log and show what loaded.
    integrations: list[IntegrationStatusOut] = []
    try:
        integrations = [
            IntegrationStatusOut(
                server_name=s.server_name,
                label=s.label,
                type=s.type,
                status=s.status,
                tool_names=s.tool_names,
            )
            for s in await AgentService.get_integration_status(agent, user=request.user)
        ]
    except Exception:
        logger.exception("Failed to load integration status for agent %s", agent_id)
    # get_tools() includes the integrations' tools; they are listed under integrations.
    integration_tools = {name for s in integrations for name in s.tool_names}
    tools: list[Tool] = []
    try:
        tools = [
            Tool(
                label=getattr(t, "label", None) or t.name.replace("_", " ").title(),
                description=t.description or "",
            )
            for t in await agent.get_tools()
            if getattr(t, "name", None) not in integration_tools
        ]
    except Exception:
        logger.exception("Failed to build tools for agent %s", agent_id)
    return ToolsResponse(tools=tools, integrations=integrations)


@routes.post("/agents/{agent_id}/run/", response=RunResponse)
async def run_agent(request: HttpRequest, agent_id: str, payload: ChatRequest) -> Any:
    """Stateless run: no thread, the whole reply at once."""
    agent = await AgentService.get(agent_id)
    await AgentService.has_perms(request.user, Operation.CHAT, agent=agent)
    chat_messages = agent.protocol_handler.to_chat_messages(payload.messages)
    result = await agent.run(chat_messages, user=request.user)
    return RunResponse(result=result, thread_id="")


@routes.post("/agents/{agent_id}/reindex/", response=Success)
async def reindex_agent(
    request: HttpRequest, agent_id: str, memory_id: str | None = None, force_rebuild: bool = False
) -> Any:
    agent = await AgentService.get(agent_id)
    # Rebuilding an index is administering the agent, not using it.
    await AgentService.has_perms(
        request.user,
        Operation.UPDATE_AGENT,
        obj=agent.config if agent.is_runtime else None,
        agent=agent,
    )
    if not await Agent.reindex(agent, memory_id, force_rebuild):
        return Success(success=False, message="No RAG provider configured for this agent")
    message = "RAG pipeline reindexed successfully" + (" (force rebuild)" if force_rebuild else "")
    if memory_id:
        message += f" for memory {memory_id}"
    return Success(success=True, message=message)


def get_agents_router(*, exclude: Collection[str] = (), **router_kwargs: Any) -> Router:
    """A new Router with the agent endpoints; mount with ``api.add_router("/", ...)``."""
    return routes.build(exclude=exclude, **router_kwargs)
