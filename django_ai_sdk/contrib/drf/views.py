"""DRF viewsets over the SDK services. Register them yourself or include
``django_ai_sdk.contrib.drf.urls``; subclass one to change a single action."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, cast

from django.urls import reverse
from pydantic import BaseModel
from rest_framework.decorators import action
from rest_framework.response import Response

from django_ai_sdk import Agent
from django_ai_sdk.agents.config import get_runtime_agent_bases, get_tool_registry
from django_ai_sdk.agents.services import AgentService, AgentUpdateData
from django_ai_sdk.contrib.drf.base import SDKViewSet, to_data
from django_ai_sdk.contrib.drf.serializers import (
    AgentSettingsSerializer,
    GroupMemberSerializer,
    MemberSerializer,
    WorkflowRunDetailSerializer,
    WorkflowRunSerializer,
    WorkflowSerializer,
)
from django_ai_sdk.errors import NotFound
from django_ai_sdk.integrations.services import IntegrationService
from django_ai_sdk.logger import get_logger
from django_ai_sdk.memories.services import MemoryService
from django_ai_sdk.permissions import Operation, PermissionDenied
from django_ai_sdk.storage.services import ThreadService, aget_thread_file_meta, aget_thread_history
from django_ai_sdk.tracing.services import TraceService
from django_ai_sdk.views.permissions import (
    aagent_permissions,
    amemory_permissions,
    athread_permissions,
)
from django_ai_sdk.views.schemas import (
    AddAgentGroupIn,
    AddAgentUserIn,
    AddMemoryGroupIn,
    AddMemoryUserIn,
    AgentSettingsCreateIn,
    AgentSettingsUpdateIn,
    BulkConnectMemoriesIn,
    ChatRequest,
    MemoryIn,
    PatchThreadPayload,
    RateMessagePayload,
    ToggleMemoryActiveIn,
    UpdateAgentUserIn,
    UpdateMemoryUserIn,
    WorkflowCreateRequest,
    WorkflowRunByIdRequest,
    WorkflowRunRequest,
    WorkflowUpdateRequest,
)
from django_ai_sdk.workflows.services import WorkflowService

if TYPE_CHECKING:
    from rest_framework.request import Request

logger = get_logger(__name__)

NO_CONTENT = 204
ID = r"[^/]+"


class TraceQuery(BaseModel):
    message_id: str | None = None
    operation_name: str | None = None
    limit: int = 100
    offset: int = 0


class ThreadViewSet(SDKViewSet):
    """Threads, their messages, traces, linked memories and uploaded files.

    Chat (streaming) is ``ChatView`` at ``threads/<id>/chat/``, see ``urls.py``.
    """

    lookup_field = "thread_id"
    lookup_value_regex = ID

    def list(self, request: Request) -> Response:
        page = self.page(request)
        threads = self.call(
            ThreadService.threads, user=request.user, limit=page.limit, offset=page.offset
        )
        return Response(to_data(threads))

    def create(self, request: Request) -> Response:
        agent_id = self.payload(ChatRequest, request.data).agent_id or ""
        thread_id = self.call(ThreadService.create_thread, agent_id=agent_id, user=request.user)
        self.call(MemoryService.link_memories, agent_id, thread_id, user=request.user)
        return Response({"thread_id": thread_id}, status=201)

    def retrieve(self, request: Request, thread_id: str) -> Response:
        data = self.call(aget_thread_history, thread_id, user=request.user)
        # Each caller sees only their own feedback on a message.
        user_pk = str(request.user.pk) if request.user.is_authenticated else None
        for message in data.get("messages", []):
            feedbacks = message.pop("feedbacks", [])
            message["feedback"] = next(
                (fb for fb in feedbacks if fb.get("user_id") == user_pk), None
            )
        data["permissions"] = self.call(athread_permissions, request.user, thread_id)
        return Response(to_data(data))

    def partial_update(self, request: Request, thread_id: str) -> Response:
        """Switch the thread to another agent, moving the linked memories along."""
        agent_id = self.payload(PatchThreadPayload, request.data).agent_id
        self.call(AgentService.get, agent_id)
        thread = self.call(ThreadService.get_thread, thread_id, user=request.user)
        if thread is None:
            raise NotFound("Thread not found")
        if thread.agent_id:
            self.call(MemoryService.unlink_memories, thread.agent_id, thread_id, user=request.user)
        self.call(
            ThreadService.update_thread,
            thread_id,
            metadata={"agent_id": agent_id},
            user=request.user,
        )
        self.call(MemoryService.link_memories, agent_id, thread_id, user=request.user)
        return Response(to_data(self.call(ThreadService.get_thread, thread_id, user=request.user)))

    def destroy(self, request: Request, thread_id: str) -> Response:
        if not self.call(ThreadService.delete_thread, thread_id, user=request.user):
            raise NotFound("Thread not found")
        return Response(status=NO_CONTENT)

    @action(detail=False, methods=["delete"], url_path="all")
    def delete_all(self, request: Request) -> Response:
        deleted = self.call(ThreadService.delete_all_threads, user=request.user)
        return Response({"deleted_count": deleted})

    @action(detail=True, methods=["post"])
    def run(self, request: Request, thread_id: str) -> Response:
        """Run the thread's agent to completion and return the whole reply."""
        payload = self.payload(ChatRequest, request.data)
        agent = self.call(AgentService.get_agent, thread_id, user=request.user)
        messages = agent.protocol_handler.to_chat_messages(payload.messages)
        result = self.call(agent.run, messages, thread_id=thread_id, user=request.user)
        return Response({"result": result, "thread_id": thread_id})

    @action(detail=True, methods=["get"], url_path="file-meta")
    def file_meta(self, request: Request, thread_id: str) -> Response:
        return Response(self.call(aget_thread_file_meta, thread_id, user=request.user))

    @action(detail=True, methods=["get"])
    def traces(self, request: Request, thread_id: str) -> Response:
        query = self.payload(TraceQuery, request.query_params.dict())
        traces = self.call(
            TraceService.thread_traces, thread_id, user=request.user, **query.model_dump()
        )
        return Response(to_data(traces))

    @action(detail=True, methods=["get"])
    def tokens(self, request: Request, thread_id: str) -> Response:
        return Response(
            to_data(self.call(TraceService.thread_token_usage, thread_id, user=request.user))
        )

    @action(detail=True, methods=["post"], url_path=rf"messages/(?P<message_id>{ID})/rate")
    def rate_message(self, request: Request, thread_id: str, message_id: str) -> Response:
        payload = self.payload(RateMessagePayload, request.data)
        self.call(
            ThreadService.rate_message,
            thread_id,
            message_id,
            payload.rating,
            feedback=payload.feedback,
            user=request.user,
        )
        return Response({"id": message_id, "is_deleted": False})

    @action(detail=True, methods=["post"], url_path=rf"messages/(?P<message_id>{ID})/delete")
    def delete_message(self, request: Request, thread_id: str, message_id: str) -> Response:
        self.call(ThreadService.delete_message, thread_id, message_id, user=request.user)
        return Response({"id": message_id, "is_deleted": True})

    @action(detail=True, methods=["post"], url_path=rf"messages/(?P<message_id>{ID})/restore")
    def restore_message(self, request: Request, thread_id: str, message_id: str) -> Response:
        self.call(ThreadService.restore_message, thread_id, message_id, user=request.user)
        return Response({"id": message_id, "is_deleted": False})

    @action(detail=True, methods=["get"])
    def memories(self, request: Request, thread_id: str) -> Response:
        page = self.page(request)
        memories = self.call(
            MemoryService.list_thread_memories,
            thread_id,
            user=request.user,
            limit=page.limit,
            offset=page.offset,
        )
        return Response(to_data(memories))

    @memories.mapping.post
    def connect_memories(self, request: Request, thread_id: str) -> Response:
        memory_ids = self.payload(BulkConnectMemoriesIn, request.data).memory_ids
        memories = self.call(
            MemoryService.bulk_connect_memories, thread_id, memory_ids, user=request.user
        )
        return Response(to_data(memories))

    @action(detail=True, methods=["patch"], url_path=rf"memories/(?P<memory_id>{ID})")
    def toggle_memory(self, request: Request, thread_id: str, memory_id: str) -> Response:
        active = self.payload(ToggleMemoryActiveIn, request.data).active
        memory = self.call(
            MemoryService.toggle_memory_active, thread_id, memory_id, active, user=request.user
        )
        return Response(to_data(memory))

    @toggle_memory.mapping.delete
    def disconnect_memory(self, request: Request, thread_id: str, memory_id: str) -> Response:
        self.call(
            MemoryService.disconnect_memory_from_thread, thread_id, memory_id, user=request.user
        )
        return Response(status=NO_CONTENT)

    @action(detail=True, methods=["get"])
    def files(self, request: Request, thread_id: str) -> Response:
        page = self.page(request)
        files = self.call(
            MemoryService.list_thread_files,
            thread_id,
            user=request.user,
            limit=page.limit,
            offset=page.offset,
        )
        return Response(to_data(files))

    @files.mapping.post
    def upload_file(self, request: Request, thread_id: str) -> Response:
        uploaded = self.call(
            MemoryService.upload_thread_file, thread_id, request.FILES["file"], user=request.user
        )
        return Response(to_data(uploaded), status=202)

    @action(detail=True, methods=["delete"], url_path=rf"files/(?P<doc_id>{ID})")
    def delete_file(self, request: Request, thread_id: str, doc_id: str) -> Response:
        self.call(MemoryService.delete_thread_file, thread_id, doc_id, user=request.user)
        return Response(status=NO_CONTENT)

    @action(detail=True, methods=["get"], url_path=rf"files/(?P<doc_id>{ID})/status")
    def file_status(self, request: Request, thread_id: str, doc_id: str) -> Response:
        return Response(
            to_data(self.call(MemoryService.get_document_status, doc_id, user=request.user))
        )


class MessageViewSet(SDKViewSet):
    """Traces and token usage for one assistant message."""

    lookup_field = "message_id"
    lookup_value_regex = ID

    @action(detail=True, methods=["get"])
    def traces(self, request: Request, message_id: str) -> Response:
        query = self.payload(TraceQuery, request.query_params.dict())
        traces = self.call(
            TraceService.message_traces,
            message_id,
            user=request.user,
            **query.model_dump(exclude={"message_id"}),
        )
        return Response(to_data(traces))

    @action(detail=True, methods=["get"])
    def tokens(self, request: Request, message_id: str) -> Response:
        return Response(
            to_data(self.call(TraceService.message_token_usage, message_id, user=request.user))
        )


class AgentViewSet(SDKViewSet):
    """Every agent the caller may use, code-declared or runtime."""

    lookup_field = "agent_id"
    lookup_value_regex = ID

    def list(self, request: Request) -> Response:
        page = self.page(request)
        agents = self.call(
            AgentService.list_agents, user=request.user, limit=page.limit, offset=page.offset
        )
        return Response(agents)

    def retrieve(self, request: Request, agent_id: str) -> Response:
        agent = self.call(AgentService.get, agent_id)
        info = to_data(self.call(AgentService.get_agent_info, agent_id, user=request.user))
        info["instructions"] = agent.get_system_prompt()
        info["permissions"] = to_data(self.call(aagent_permissions, request.user, agent_id))
        return Response(info)

    @action(detail=True, methods=["get"])
    def tools(self, request: Request, agent_id: str) -> Response:
        agent = self.call(AgentService.get, agent_id)
        self._check(request, agent, Operation.VIEW_AGENT)
        # A broken tool or integration must not hide the rest: log and show what loaded.
        tools: list[dict[str, str]] = []
        try:
            tools = [
                {
                    "label": getattr(t, "label", None) or t.name.replace("_", " ").title(),
                    "description": t.description or "",
                }
                for t in self.call(agent.get_tools)
            ]
        except Exception:
            logger.exception("Failed to build tools for agent %s", agent_id)
        integrations: list[Any] = []
        try:
            integrations = self.call(AgentService.get_integration_status, agent, user=request.user)
        except Exception:
            logger.exception("Failed to load integration status for agent %s", agent_id)
        return Response({"tools": tools, "integrations": to_data(integrations)})

    @action(detail=True, methods=["post"])
    def run(self, request: Request, agent_id: str) -> Response:
        """Stateless run: no thread, the whole reply at once."""
        payload = self.payload(ChatRequest, request.data)
        agent = self.call(AgentService.get, agent_id)
        messages = agent.protocol_handler.to_chat_messages(payload.messages)
        return Response({"result": self.call(agent.run, messages, user=request.user)})

    @action(detail=True, methods=["post"])
    def reindex(self, request: Request, agent_id: str) -> Response:
        agent = self.call(AgentService.get, agent_id)
        # Rebuilding an index is administering the agent, not using it.
        self._check(request, agent, Operation.UPDATE_AGENT)
        memory_id = request.query_params.get("memory_id")
        force = request.query_params.get("force_rebuild", "").lower() in {"1", "true", "yes"}
        return Response({"success": bool(self.call(Agent.reindex, agent, memory_id, force))})

    def _check(self, request: Request, agent: Any, operation: Operation) -> None:
        self.call(
            AgentService.has_perms,
            request.user,
            operation,
            obj=agent.config if agent.is_runtime else None,
            agent=agent,
        )


class RuntimeAgentViewSet(SDKViewSet):
    """Agents configured in the database, and who they are shared with."""

    lookup_field = "runtime_id"
    lookup_value_regex = ID

    def list(self, request: Request) -> Response:
        configs = self.call(AgentService.list_runtime_agents, user=request.user)
        return Response(AgentSettingsSerializer(configs, many=True).data)

    def create(self, request: Request) -> Response:
        payload = self.payload(AgentSettingsCreateIn, request.data)
        config = self.call(
            AgentService.create_runtime_agent,
            payload.model_dump(exclude={"users", "groups"}),
            user=request.user,
        )
        # Sharing is best effort: the agent exists either way, and the owner can retry.
        for entry in payload.users:
            with contextlib.suppress(PermissionDenied, ValueError):
                self.call(
                    AgentService.add_agent_user,
                    str(config.id),
                    entry.user_id,
                    entry.can_manage,
                    user=request.user,
                )
        for group in payload.groups:
            with contextlib.suppress(PermissionDenied, ValueError):
                self.call(
                    AgentService.add_agent_group,
                    str(config.id),
                    group.group_id,
                    group.can_manage,
                    user=request.user,
                )
        return Response(AgentSettingsSerializer(config).data, status=201)

    def retrieve(self, request: Request, runtime_id: str) -> Response:
        config = self.call(AgentService.get_runtime_agent, runtime_id, user=request.user)
        return Response(AgentSettingsSerializer(config).data)

    def partial_update(self, request: Request, runtime_id: str) -> Response:
        payload = self.payload(AgentSettingsUpdateIn, request.data)
        data = cast("AgentUpdateData", payload.model_dump(exclude_none=True))
        config = self.call(AgentService.update_runtime_agent, runtime_id, data, user=request.user)
        return Response(AgentSettingsSerializer(config).data)

    def destroy(self, request: Request, runtime_id: str) -> Response:
        self.call(AgentService.delete_runtime_agent, runtime_id, user=request.user)
        return Response(status=NO_CONTENT)

    @action(detail=False, methods=["get"])
    def bases(self, request: Request) -> Response:
        return Response(
            [
                {"path": f"{cls.__module__}.{cls.__qualname__}", "name": cls.__name__}
                for cls in get_runtime_agent_bases()
            ]
        )

    @action(detail=False, methods=["get"], url_path="tools")
    def available_tools(self, request: Request) -> Response:
        return Response([{"key": k, "path": p} for k, p in get_tool_registry().items()])

    @action(detail=True, methods=["get"])
    def users(self, request: Request, runtime_id: str) -> Response:
        users = self.call(AgentService.list_agent_users, runtime_id, user=request.user)
        return Response(MemberSerializer(users, many=True).data)

    @users.mapping.post
    def add_user(self, request: Request, runtime_id: str) -> Response:
        payload = self.payload(AddAgentUserIn, request.data)
        entry = self.call(
            AgentService.add_agent_user,
            runtime_id,
            payload.user_id,
            payload.can_manage,
            user=request.user,
        )
        return Response(MemberSerializer(entry).data, status=201)

    @action(detail=True, methods=["patch"], url_path=rf"users/(?P<user_id>{ID})")
    def update_user(self, request: Request, runtime_id: str, user_id: str) -> Response:
        can_manage = self.payload(UpdateAgentUserIn, request.data).can_manage
        entry = self.call(
            AgentService.update_agent_user, runtime_id, user_id, can_manage, user=request.user
        )
        return Response(MemberSerializer(entry).data)

    @update_user.mapping.delete
    def remove_user(self, request: Request, runtime_id: str, user_id: str) -> Response:
        self.call(AgentService.remove_agent_user, runtime_id, user_id, user=request.user)
        return Response(status=NO_CONTENT)

    @action(detail=True, methods=["get"])
    def groups(self, request: Request, runtime_id: str) -> Response:
        groups = self.call(AgentService.list_agent_groups, runtime_id, user=request.user)
        return Response(GroupMemberSerializer(groups, many=True).data)

    @groups.mapping.post
    def add_group(self, request: Request, runtime_id: str) -> Response:
        payload = self.payload(AddAgentGroupIn, request.data)
        entry = self.call(
            AgentService.add_agent_group,
            runtime_id,
            payload.group_id,
            payload.can_manage,
            user=request.user,
        )
        return Response(GroupMemberSerializer(entry).data, status=201)

    @action(detail=True, methods=["delete"], url_path=r"groups/(?P<group_id>\d+)")
    def remove_group(self, request: Request, runtime_id: str, group_id: str) -> Response:
        self.call(AgentService.remove_agent_group, runtime_id, int(group_id), user=request.user)
        return Response(status=NO_CONTENT)


class MemoryViewSet(SDKViewSet):
    """Memories (knowledge bases), their documents, and who they are shared with."""

    lookup_field = "memory_id"
    lookup_value_regex = ID

    def list(self, request: Request) -> Response:
        page = self.page(request)
        memories = self.call(
            MemoryService.list_memories, user=request.user, limit=page.limit, offset=page.offset
        )
        return Response([self._with_permissions(request, m) for m in memories])

    def create(self, request: Request) -> Response:
        payload = self.payload(MemoryIn, request.data)
        memory = self.call(MemoryService.create_memory, **payload.model_dump(), user=request.user)
        return Response(to_data(memory), status=201)

    def retrieve(self, request: Request, memory_id: str) -> Response:
        memory = self.call(MemoryService.get_memory, memory_id, user=request.user)
        return Response(self._with_permissions(request, memory))

    def update(self, request: Request, memory_id: str) -> Response:
        payload = self.payload(MemoryIn, request.data)
        memory = self.call(
            MemoryService.update_memory,
            memory_id=memory_id,
            name=payload.name,
            description=payload.description,
            is_public=payload.is_public,
            user=request.user,
        )
        return Response(to_data(memory))

    def destroy(self, request: Request, memory_id: str) -> Response:
        self.call(MemoryService.delete_memory, memory_id, user=request.user)
        return Response(status=NO_CONTENT)

    # Not named `settings`: that would shadow APIView.settings.
    @action(detail=False, methods=["get"], url_path="settings")
    def upload_settings(self, request: Request) -> Response:
        from django_ai_sdk import files

        return Response(to_data(files.get_upload_settings()))

    @action(
        detail=False, methods=["get"], url_path=rf"source/(?P<entry_id>{ID})/(?P<chunk_id>{ID})"
    )
    def source(self, request: Request, entry_id: str, chunk_id: str) -> Response:
        content = self.call(MemoryService.get_chunk_content, entry_id, chunk_id, user=request.user)
        if content is None:
            raise NotFound(f"Entry not found: {entry_id}")
        return Response({"content": content})

    @action(detail=True, methods=["get"])
    def documents(self, request: Request, memory_id: str) -> Response:
        page = self.page(request)
        docs = self.call(
            MemoryService.list_documents,
            memory_id,
            user=request.user,
            limit=page.limit,
            offset=page.offset,
        )
        return Response(to_data(docs))

    @documents.mapping.post
    def upload_document(self, request: Request, memory_id: str) -> Response:
        uploaded = self.call(
            MemoryService.upload_document, memory_id, request.FILES["file"], user=request.user
        )
        return Response(to_data(uploaded), status=202)

    @action(detail=True, methods=["get"], url_path=rf"documents/(?P<doc_id>{ID})")
    def document(self, request: Request, memory_id: str, doc_id: str) -> Response:
        return Response(
            to_data(self.call(MemoryService.get_document, memory_id, doc_id, user=request.user))
        )

    @document.mapping.delete
    def delete_document(self, request: Request, memory_id: str, doc_id: str) -> Response:
        self.call(MemoryService.delete_document, memory_id, doc_id, user=request.user)
        return Response(status=NO_CONTENT)

    @action(detail=True, methods=["get"], url_path=rf"documents/(?P<doc_id>{ID})/status")
    def document_status(self, request: Request, memory_id: str, doc_id: str) -> Response:
        return Response(
            to_data(self.call(MemoryService.get_document_status, doc_id, user=request.user))
        )

    @action(detail=True, methods=["post"], url_path=rf"threads/(?P<thread_id>{ID})")
    def link_thread(self, request: Request, memory_id: str, thread_id: str) -> Response:
        self.call(MemoryService.link_memory_to_thread, memory_id, thread_id, user=request.user)
        return Response(status=NO_CONTENT)

    @link_thread.mapping.delete
    def unlink_thread(self, request: Request, memory_id: str, thread_id: str) -> Response:
        self.call(MemoryService.unlink_memory_from_thread, memory_id, thread_id, user=request.user)
        return Response(status=NO_CONTENT)

    @action(detail=True, methods=["get"])
    def users(self, request: Request, memory_id: str) -> Response:
        page = self.page(request)
        users = self.call(
            MemoryService.list_memory_users,
            memory_id,
            user=request.user,
            limit=page.limit,
            offset=page.offset,
        )
        return Response(to_data(users))

    @users.mapping.post
    def add_user(self, request: Request, memory_id: str) -> Response:
        payload = self.payload(AddMemoryUserIn, request.data)
        entry = self.call(
            MemoryService.add_memory_user,
            memory_id,
            payload.user_id,
            payload.can_manage,
            user=request.user,
        )
        return Response(to_data(entry), status=201)

    @action(detail=True, methods=["patch"], url_path=rf"users/(?P<user_id>{ID})")
    def update_user(self, request: Request, memory_id: str, user_id: str) -> Response:
        can_manage = self.payload(UpdateMemoryUserIn, request.data).can_manage
        entry = self.call(
            MemoryService.update_memory_user, memory_id, user_id, can_manage, user=request.user
        )
        return Response(to_data(entry))

    @update_user.mapping.delete
    def remove_user(self, request: Request, memory_id: str, user_id: str) -> Response:
        self.call(MemoryService.remove_memory_user, memory_id, user_id, user=request.user)
        return Response(status=NO_CONTENT)

    @action(detail=True, methods=["get"])
    def groups(self, request: Request, memory_id: str) -> Response:
        groups = self.call(MemoryService.list_memory_groups, memory_id, user=request.user)
        return Response(to_data(groups))

    @groups.mapping.post
    def add_group(self, request: Request, memory_id: str) -> Response:
        payload = self.payload(AddMemoryGroupIn, request.data)
        entry = self.call(
            MemoryService.add_memory_group,
            memory_id,
            payload.group_id,
            payload.can_manage,
            user=request.user,
        )
        return Response(to_data(entry), status=201)

    @action(detail=True, methods=["delete"], url_path=r"groups/(?P<group_id>\d+)")
    def remove_group(self, request: Request, memory_id: str, group_id: str) -> Response:
        self.call(MemoryService.remove_memory_group, memory_id, int(group_id), user=request.user)
        return Response(status=NO_CONTENT)

    def _with_permissions(self, request: Request, memory: Any) -> dict[str, Any]:
        perms = self.call(amemory_permissions, request.user, memory.id)
        return {**to_data(memory), "permissions": to_data(perms)}


class WorkflowViewSet(SDKViewSet):
    """Stored workflow definitions, their runs, and ad-hoc runs."""

    lookup_field = "workflow_id"
    lookup_value_regex = ID

    def list(self, request: Request) -> Response:
        page = self.page(request)
        records = self.call(
            WorkflowService.list_workflows,
            user=request.user,
            limit=page.limit,
            offset=page.offset,
        )
        return Response(WorkflowSerializer(records, many=True).data)

    def create(self, request: Request) -> Response:
        payload = self.payload(WorkflowCreateRequest, request.data)
        record = self.call(
            WorkflowService.create, payload.name, payload.workflow, user=request.user
        )
        return Response(WorkflowSerializer(record).data, status=201)

    def retrieve(self, request: Request, workflow_id: str) -> Response:
        record = self.call(WorkflowService.get, workflow_id, user=request.user)
        return Response(WorkflowSerializer(record).data)

    def partial_update(self, request: Request, workflow_id: str) -> Response:
        payload = self.payload(WorkflowUpdateRequest, request.data)
        record = self.call(
            WorkflowService.update,
            workflow_id,
            user=request.user,
            name=payload.name,
            workflow=payload.workflow,
            active=payload.active,
        )
        return Response(WorkflowSerializer(record).data)

    def destroy(self, request: Request, workflow_id: str) -> Response:
        self.call(WorkflowService.delete, workflow_id, user=request.user)
        return Response(status=NO_CONTENT)

    @action(detail=False, methods=["post"], url_path="run")
    def run_definition(self, request: Request) -> Response:
        """Run a definition sent in the request, without storing it."""
        payload = self.payload(WorkflowRunRequest, request.data)
        run = self.call(
            WorkflowService.run, payload.workflow, inputs=payload.inputs, user=request.user
        )
        return Response({"run_id": str(run.id), "status": run.status}, status=202)

    @action(detail=False, methods=["get"], url_path="actions")
    def list_actions(self, request: Request) -> Response:
        return Response(WorkflowService.list_actions())

    @action(detail=True, methods=["post"])
    def run(self, request: Request, workflow_id: str) -> Response:
        payload = self.payload(WorkflowRunByIdRequest, request.data)
        run = self.call(
            WorkflowService.run_by_id,
            workflow_id,
            inputs=payload.inputs,
            user=request.user,
            run_id=payload.run_id,
        )
        return Response({"run_id": str(run.id), "status": run.status}, status=202)

    @action(detail=True, methods=["get"])
    def runs(self, request: Request, workflow_id: str) -> Response:
        page = self.page(request, limit=50)
        runs = self.call(
            WorkflowService.list_runs,
            workflow_id,
            user=request.user,
            limit=page.limit,
            offset=page.offset,
        )
        return Response(WorkflowRunSerializer(runs, many=True).data)

    @action(detail=True, methods=["get"], url_path=rf"runs/(?P<run_id>{ID})")
    def run_detail(self, request: Request, workflow_id: str, run_id: str) -> Response:
        run = self.call(WorkflowService.get_run, run_id, user=request.user)
        if run is None:
            # Absent covers both "no such run" and "not yours", by design.
            raise NotFound("Run not found")
        return Response(WorkflowRunDetailSerializer(run).data)


class IntegrationViewSet(SDKViewSet):
    """Integrations the caller may use, with live status; connect/disconnect/reconnect.

    The OAuth callback is a plain Django view at a fixed URL:
    ``include("django_ai_sdk.integrations.mcp.urls")``.
    """

    lookup_field = "name"
    lookup_value_regex = ID

    def list(self, request: Request) -> Response:
        return Response(to_data(self.call(IntegrationService.list_for_user, request.user)))

    @action(detail=True, methods=["post"])
    def connect(self, request: Request, name: str) -> Response:
        redirect_uri = request.build_absolute_uri(
            reverse("integrations_mcp:oauth-callback", kwargs={"server_name": name})
        )
        result = self.call(
            IntegrationService.connect,
            name,
            request.user,
            request=request,
            redirect_uri=redirect_uri,
        )
        if result is None:
            raise NotFound("Unknown integration")
        return Response({"redirect_url": result["redirect_url"]})

    @action(detail=True, methods=["post"])
    def disconnect(self, request: Request, name: str) -> Response:
        deleted = self.call(IntegrationService.disconnect, name, request.user)
        if deleted is None:
            raise NotFound("Unknown integration")
        if not deleted:
            raise NotFound("Not connected")
        return Response(status=NO_CONTENT)

    @action(detail=True, methods=["post"])
    def reconnect(self, request: Request, name: str) -> Response:
        status = self.call(IntegrationService.reconnect, name, request.user)
        if status is None:
            raise NotFound("Unknown integration")
        return Response({"status": status})
