"""DRF viewsets over the SDK services. Register them yourself or include
``django_ai_sdk.contrib.drf.urls``; subclass one to change a single action.

DRF views are sync, so they call the services' sync wrappers
(``django_ai_sdk.storage.services.list_threads`` and friends).
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, cast

from asgiref.sync import async_to_sync
from django.urls import reverse
from rest_framework.decorators import action
from rest_framework.response import Response

from django_ai_sdk import Agent
from django_ai_sdk.agents import services as agent_services
from django_ai_sdk.agents.config import get_runtime_agent_bases, get_tool_registry
from django_ai_sdk.agents.services import AgentCreateData, AgentService, AgentUpdateData
from django_ai_sdk.contrib.drf import serializers as s
from django_ai_sdk.contrib.drf.base import ApiViewSet
from django_ai_sdk.errors import NotFound
from django_ai_sdk.integrations import services as integration_services
from django_ai_sdk.logger import get_logger
from django_ai_sdk.memories import services as memory_services
from django_ai_sdk.permissions import Operation, PermissionDenied
from django_ai_sdk.storage import services as thread_services
from django_ai_sdk.tracing import services as trace_services
from django_ai_sdk.views import permissions as object_permissions
from django_ai_sdk.views.files import thread_file_response
from django_ai_sdk.views.schemas import Message
from django_ai_sdk.workflows import services as workflow_services
from django_ai_sdk.workflows.services import WorkflowService

if TYPE_CHECKING:
    from django.http import FileResponse
    from rest_framework.request import Request

logger = get_logger(__name__)

NO_CONTENT = 204
ID = r"[^/]+"

get_agent = async_to_sync(AgentService.get)
get_thread_agent = async_to_sync(AgentService.get_agent)
check_agent_perms = async_to_sync(AgentService.has_perms)


def chat_messages(agent: Agent, validated: dict[str, Any]) -> list[Any]:
    """Validated request messages in the agent's internal format."""
    return agent.protocol_handler.to_chat_messages(
        [Message.model_validate(m) for m in validated["messages"]]
    )


class ThreadViewSet(ApiViewSet):
    """Threads, their messages, traces, linked memories and uploaded files.

    Chat (streaming) is ``ChatView`` at ``threads/<id>/chat/``, see ``urls.py``.
    """

    lookup_field = "thread_id"
    lookup_value_regex = ID
    serializer_classes = {
        "create": s.ChatRequestSerializer,
        "partial_update": s.AgentSwitchSerializer,
        "run": s.ChatRequestSerializer,
        "rate_message": s.RateMessageSerializer,
        "connect_memories": s.BulkConnectSerializer,
        "toggle_memory": s.ToggleActiveSerializer,
    }

    def list(self, request: Request) -> Response:
        limit, offset = self.page(request)
        threads = thread_services.list_threads(user=request.user, limit=limit, offset=offset)
        return Response(s.ThreadSerializer(threads, many=True).data)

    def create(self, request: Request) -> Response:
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        agent_id = serializer.validated_data.get("agent_id") or ""
        thread_id = thread_services.create_thread(agent_id=agent_id, user=request.user)
        memory_services.link_memories(agent_id, thread_id, user=request.user)
        return Response({"thread_id": thread_id}, status=201)

    def retrieve(self, request: Request, thread_id: str) -> Response:
        data = thread_services.get_thread_history(thread_id, user=request.user)
        # Each caller sees only their own feedback on a message.
        user_pk = str(request.user.pk) if request.user.is_authenticated else None
        for message in data.get("messages", []):
            feedbacks = message.pop("feedbacks", [])
            message["feedback"] = next(
                (fb for fb in feedbacks if fb.get("user_id") == user_pk), None
            )
        data["permissions"] = object_permissions.thread_permissions(request.user, thread_id)
        return Response(s.ThreadDetailSerializer(data).data)

    def partial_update(self, request: Request, thread_id: str) -> Response:
        """Switch the thread to another agent, moving the linked memories along."""
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        agent_id = serializer.validated_data["agent_id"]
        get_agent(agent_id)
        thread = thread_services.get_thread(thread_id, user=request.user)
        if thread is None:
            raise NotFound("Thread not found")
        if thread.agent_id:
            memory_services.unlink_memories(thread.agent_id, thread_id, user=request.user)
        thread_services.update_thread(thread_id, metadata={"agent_id": agent_id}, user=request.user)
        memory_services.link_memories(agent_id, thread_id, user=request.user)
        thread = thread_services.get_thread(thread_id, user=request.user)
        return Response(s.ThreadSerializer(thread).data)

    def destroy(self, request: Request, thread_id: str) -> Response:
        if not thread_services.delete_thread(thread_id, user=request.user):
            raise NotFound("Thread not found")
        return Response(status=NO_CONTENT)

    @action(detail=False, methods=["delete"], url_path="all")
    def delete_all(self, request: Request) -> Response:
        deleted = thread_services.delete_all_threads(user=request.user)
        return Response({"deleted_count": deleted})

    @action(detail=True, methods=["post"])
    def run(self, request: Request, thread_id: str) -> Response:
        """Run the thread's agent to completion and return the whole reply."""
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        agent = get_thread_agent(thread_id, user=request.user)
        messages = chat_messages(agent, serializer.validated_data)
        result = async_to_sync(agent.run)(messages, thread_id=thread_id, user=request.user)
        return Response({"result": result, "thread_id": thread_id})

    @action(detail=True, methods=["get"], url_path="file-meta")
    def file_meta(self, request: Request, thread_id: str) -> Response:
        meta = thread_services.get_thread_file_meta(thread_id, user=request.user)
        return Response(s.ThreadFileMetaSerializer(meta).data)

    @action(detail=True, methods=["get"])
    def traces(self, request: Request, thread_id: str) -> Response:
        query = s.TraceQuerySerializer(data=request.query_params)
        query.is_valid(raise_exception=True)
        limit, offset = self.page(request)
        traces = trace_services.thread_traces(
            thread_id, user=request.user, limit=limit, offset=offset, **query.validated_data
        )
        return Response(s.TraceSerializer(traces, many=True).data)

    @action(detail=True, methods=["get"])
    def tokens(self, request: Request, thread_id: str) -> Response:
        usage = trace_services.thread_token_usage(thread_id, user=request.user)
        return Response(s.TokenUsageSerializer(usage).data)

    @action(detail=True, methods=["post"], url_path=rf"messages/(?P<message_id>{ID})/rate")
    def rate_message(self, request: Request, thread_id: str, message_id: str) -> Response:
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        thread_services.rate_message(
            thread_id,
            message_id,
            serializer.validated_data.get("rating"),
            feedback=serializer.validated_data["feedback"],
            user=request.user,
        )
        return Response({"id": message_id, "is_deleted": False})

    @action(detail=True, methods=["post"], url_path=rf"messages/(?P<message_id>{ID})/delete")
    def delete_message(self, request: Request, thread_id: str, message_id: str) -> Response:
        thread_services.delete_message(thread_id, message_id, user=request.user)
        return Response({"id": message_id, "is_deleted": True})

    @action(detail=True, methods=["post"], url_path=rf"messages/(?P<message_id>{ID})/restore")
    def restore_message(self, request: Request, thread_id: str, message_id: str) -> Response:
        thread_services.restore_message(thread_id, message_id, user=request.user)
        return Response({"id": message_id, "is_deleted": False})

    @action(detail=True, methods=["get"])
    def memories(self, request: Request, thread_id: str) -> Response:
        limit, offset = self.page(request)
        memories = memory_services.list_thread_memories(
            thread_id, user=request.user, limit=limit, offset=offset
        )
        return Response(s.ThreadMemorySerializer(memories, many=True).data)

    @memories.mapping.post
    def connect_memories(self, request: Request, thread_id: str) -> Response:
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        memories = memory_services.bulk_connect_memories(
            thread_id, serializer.validated_data["memory_ids"], user=request.user
        )
        return Response(s.ThreadMemorySerializer(memories, many=True).data)

    @action(detail=True, methods=["patch"], url_path=rf"memories/(?P<memory_id>{ID})")
    def toggle_memory(self, request: Request, thread_id: str, memory_id: str) -> Response:
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        memory = memory_services.toggle_memory_active(
            thread_id, memory_id, serializer.validated_data["active"], user=request.user
        )
        return Response(s.ThreadMemorySerializer(memory).data)

    @toggle_memory.mapping.delete
    def disconnect_memory(self, request: Request, thread_id: str, memory_id: str) -> Response:
        memory_services.disconnect_memory_from_thread(thread_id, memory_id, user=request.user)
        return Response(status=NO_CONTENT)

    @action(detail=True, methods=["get"])
    def files(self, request: Request, thread_id: str) -> Response:
        limit, offset = self.page(request)
        files = memory_services.list_thread_files(
            thread_id, user=request.user, limit=limit, offset=offset
        )
        return Response(s.DocumentSerializer(files, many=True).data)

    @files.mapping.post
    def upload_file(self, request: Request, thread_id: str) -> Response:
        uploaded = memory_services.upload_thread_file(
            thread_id, request.FILES["file"], user=request.user
        )
        return Response(s.UploadResultSerializer(uploaded).data, status=202)

    @action(detail=True, methods=["delete"], url_path=rf"files/(?P<doc_id>{ID})")
    def delete_file(self, request: Request, thread_id: str, doc_id: str) -> Response:
        memory_services.delete_thread_file(thread_id, doc_id, user=request.user)
        return Response(status=NO_CONTENT)

    @action(detail=True, methods=["get"], url_path=rf"files/(?P<doc_id>{ID})/download")
    def download_file(self, request: Request, thread_id: str, doc_id: str) -> FileResponse:
        """The file's bytes. Only raster images render inline; everything else downloads."""
        doc = memory_services.get_thread_file(thread_id, doc_id, user=request.user)
        return thread_file_response(doc)

    @action(detail=True, methods=["get"], url_path=rf"files/(?P<doc_id>{ID})/status")
    def file_status(self, request: Request, thread_id: str, doc_id: str) -> Response:
        status = memory_services.get_document_status(doc_id, user=request.user)
        return Response(s.DocumentStatusSerializer(status).data)


class MessageViewSet(ApiViewSet):
    """Traces and token usage for one assistant message."""

    lookup_field = "message_id"
    lookup_value_regex = ID

    @action(detail=True, methods=["get"])
    def traces(self, request: Request, message_id: str) -> Response:
        query = s.TraceQuerySerializer(data=request.query_params)
        query.is_valid(raise_exception=True)
        limit, offset = self.page(request)
        traces = trace_services.message_traces(
            message_id,
            user=request.user,
            operation_name=query.validated_data["operation_name"],
            limit=limit,
            offset=offset,
        )
        return Response(s.TraceSerializer(traces, many=True).data)

    @action(detail=True, methods=["get"])
    def tokens(self, request: Request, message_id: str) -> Response:
        usage = trace_services.message_token_usage(message_id, user=request.user)
        return Response(s.TokenUsageSerializer(usage).data)


class AgentViewSet(ApiViewSet):
    """Every agent the caller may use, code-declared or runtime."""

    lookup_field = "agent_id"
    lookup_value_regex = ID
    serializer_classes = {"run": s.ChatRequestSerializer}

    def list(self, request: Request) -> Response:
        limit, offset = self.page(request)
        agents = agent_services.list_agents(user=request.user, limit=limit, offset=offset)
        return Response(s.AgentSummarySerializer(agents, many=True).data)

    def retrieve(self, request: Request, agent_id: str) -> Response:
        agent = get_agent(agent_id)
        info = agent_services.get_agent_info(agent_id, user=request.user)
        data = {
            **info.model_dump(),
            "instructions": agent.get_system_prompt(),
            "permissions": object_permissions.agent_permissions(request.user, agent_id),
        }
        return Response(s.AgentInfoSerializer(data).data)

    @action(detail=True, methods=["get"])
    def tools(self, request: Request, agent_id: str) -> Response:
        agent = get_agent(agent_id)
        self._check(request, agent, Operation.VIEW_AGENT)
        # A broken tool or integration must not hide the rest: log and show what loaded.
        tools: list[dict[str, str]] = []
        try:
            tools = [
                {
                    "label": getattr(t, "label", None) or t.name.replace("_", " ").title(),
                    "description": t.description or "",
                }
                for t in async_to_sync(agent.get_tools)()
            ]
        except Exception:
            logger.exception("Failed to build tools for agent %s", agent_id)
        integrations: list[Any] = []
        try:
            integrations = agent_services.get_integration_status(agent, user=request.user)
        except Exception:
            logger.exception("Failed to load integration status for agent %s", agent_id)
        return Response(s.AgentToolsSerializer({"tools": tools, "integrations": integrations}).data)

    @action(detail=True, methods=["post"])
    def run(self, request: Request, agent_id: str) -> Response:
        """Stateless run: no thread, the whole reply at once."""
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        agent = get_agent(agent_id)
        messages = chat_messages(agent, serializer.validated_data)
        return Response({"result": async_to_sync(agent.run)(messages, user=request.user)})

    @action(detail=True, methods=["post"])
    def reindex(self, request: Request, agent_id: str) -> Response:
        query = s.ReindexQuerySerializer(data=request.query_params)
        query.is_valid(raise_exception=True)
        agent = get_agent(agent_id)
        # Rebuilding an index is administering the agent, not using it.
        self._check(request, agent, Operation.UPDATE_AGENT)
        result = async_to_sync(Agent.reindex)(
            agent, query.validated_data["memory_id"], query.validated_data["force_rebuild"]
        )
        return Response({"success": bool(result)})

    def _check(self, request: Request, agent: Any, operation: Operation) -> None:
        check_agent_perms(
            request.user,
            operation,
            obj=agent.config if agent.is_runtime else None,
            agent=agent,
        )


class RuntimeAgentViewSet(ApiViewSet):
    """Agents configured in the database, and who they are shared with."""

    lookup_field = "runtime_id"
    lookup_value_regex = ID
    serializer_classes = {
        "create": s.RuntimeAgentCreateSerializer,
        "partial_update": s.RuntimeAgentUpdateSerializer,
        "add_user": s.AddUserSerializer,
        "update_user": s.UpdateUserSerializer,
        "add_group": s.AddGroupSerializer,
    }

    def list(self, request: Request) -> Response:
        configs = agent_services.list_runtime_agents(user=request.user)
        return Response(s.AgentSettingsSerializer(configs, many=True).data)

    def create(self, request: Request) -> Response:
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = dict(serializer.validated_data)
        users, groups = data.pop("users"), data.pop("groups")
        config = agent_services.create_runtime_agent(
            cast("AgentCreateData", data), user=request.user
        )
        # Sharing is best effort: the agent exists either way, and the owner can retry.
        for entry in users:
            with contextlib.suppress(PermissionDenied, ValueError):
                agent_services.add_agent_user(
                    str(config.id), entry["user_id"], entry["can_manage"], user=request.user
                )
        for group in groups:
            with contextlib.suppress(PermissionDenied, ValueError):
                agent_services.add_agent_group(
                    str(config.id), group["group_id"], group["can_manage"], user=request.user
                )
        return Response(s.AgentSettingsSerializer(config).data, status=201)

    def retrieve(self, request: Request, runtime_id: str) -> Response:
        config = agent_services.get_runtime_agent(runtime_id, user=request.user)
        return Response(s.AgentSettingsSerializer(config).data)

    def partial_update(self, request: Request, runtime_id: str) -> Response:
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = cast("AgentUpdateData", dict(serializer.validated_data))
        config = agent_services.update_runtime_agent(runtime_id, data, user=request.user)
        return Response(s.AgentSettingsSerializer(config).data)

    def destroy(self, request: Request, runtime_id: str) -> Response:
        agent_services.delete_runtime_agent(runtime_id, user=request.user)
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
        users = agent_services.list_agent_users(runtime_id, user=request.user)
        return Response(s.AgentMemberSerializer(users, many=True).data)

    @users.mapping.post
    def add_user(self, request: Request, runtime_id: str) -> Response:
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        entry = agent_services.add_agent_user(
            runtime_id,
            serializer.validated_data["user_id"],
            serializer.validated_data["can_manage"],
            user=request.user,
        )
        return Response(s.AgentMemberSerializer(entry).data, status=201)

    @action(detail=True, methods=["patch"], url_path=rf"users/(?P<user_id>{ID})")
    def update_user(self, request: Request, runtime_id: str, user_id: str) -> Response:
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        entry = agent_services.update_agent_user(
            runtime_id, user_id, serializer.validated_data["can_manage"], user=request.user
        )
        return Response(s.AgentMemberSerializer(entry).data)

    @update_user.mapping.delete
    def remove_user(self, request: Request, runtime_id: str, user_id: str) -> Response:
        agent_services.remove_agent_user(runtime_id, user_id, user=request.user)
        return Response(status=NO_CONTENT)

    @action(detail=True, methods=["get"])
    def groups(self, request: Request, runtime_id: str) -> Response:
        groups = agent_services.list_agent_groups(runtime_id, user=request.user)
        return Response(s.AgentGroupMemberSerializer(groups, many=True).data)

    @groups.mapping.post
    def add_group(self, request: Request, runtime_id: str) -> Response:
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        entry = agent_services.add_agent_group(
            runtime_id,
            serializer.validated_data["group_id"],
            serializer.validated_data["can_manage"],
            user=request.user,
        )
        return Response(s.AgentGroupMemberSerializer(entry).data, status=201)

    @action(detail=True, methods=["delete"], url_path=r"groups/(?P<group_id>\d+)")
    def remove_group(self, request: Request, runtime_id: str, group_id: str) -> Response:
        agent_services.remove_agent_group(runtime_id, int(group_id), user=request.user)
        return Response(status=NO_CONTENT)


class MemoryViewSet(ApiViewSet):
    """Memories (knowledge bases), their documents, and who they are shared with."""

    lookup_field = "memory_id"
    lookup_value_regex = ID
    serializer_classes = {
        "create": s.MemorySerializer,
        "update": s.MemorySerializer,
        "add_user": s.AddUserSerializer,
        "update_user": s.UpdateUserSerializer,
        "add_group": s.AddGroupSerializer,
    }

    def list(self, request: Request) -> Response:
        limit, offset = self.page(request)
        memories = memory_services.list_memories(user=request.user, limit=limit, offset=offset)
        return Response([self._with_permissions(request, m) for m in memories])

    def create(self, request: Request) -> Response:
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        memory = memory_services.create_memory(**serializer.validated_data, user=request.user)
        return Response(s.MemoryOutSerializer(memory).data, status=201)

    def retrieve(self, request: Request, memory_id: str) -> Response:
        memory = memory_services.get_memory(memory_id, user=request.user)
        return Response(self._with_permissions(request, memory))

    def update(self, request: Request, memory_id: str) -> Response:
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        memory = memory_services.update_memory(
            memory_id=memory_id,
            name=data["name"],
            description=data["description"],
            is_public=data["is_public"],
            user=request.user,
        )
        return Response(s.MemoryOutSerializer(memory).data)

    def destroy(self, request: Request, memory_id: str) -> Response:
        memory_services.delete_memory(memory_id, user=request.user)
        return Response(status=NO_CONTENT)

    # Not named `settings`: that would shadow APIView.settings.
    @action(detail=False, methods=["get"], url_path="settings")
    def upload_settings(self, request: Request) -> Response:
        from django_ai_sdk import files

        return Response(s.UploadSettingsSerializer(files.get_upload_settings()).data)

    @action(
        detail=False, methods=["get"], url_path=rf"source/(?P<entry_id>{ID})/(?P<chunk_id>{ID})"
    )
    def source(self, request: Request, entry_id: str, chunk_id: str) -> Response:
        content = memory_services.get_chunk_content(entry_id, chunk_id, user=request.user)
        if content is None:
            raise NotFound(f"Entry not found: {entry_id}")
        return Response({"content": content})

    @action(detail=True, methods=["get"])
    def documents(self, request: Request, memory_id: str) -> Response:
        limit, offset = self.page(request)
        docs = memory_services.list_documents(
            memory_id, user=request.user, limit=limit, offset=offset
        )
        return Response(s.DocumentSerializer(docs, many=True).data)

    @documents.mapping.post
    def upload_document(self, request: Request, memory_id: str) -> Response:
        uploaded = memory_services.upload_document(
            memory_id, request.FILES["file"], user=request.user
        )
        return Response(s.UploadResultSerializer(uploaded).data, status=202)

    @action(detail=True, methods=["get"], url_path=rf"documents/(?P<doc_id>{ID})")
    def document(self, request: Request, memory_id: str, doc_id: str) -> Response:
        doc = memory_services.get_document(memory_id, doc_id, user=request.user)
        return Response(s.DocumentSerializer(doc).data)

    @document.mapping.delete
    def delete_document(self, request: Request, memory_id: str, doc_id: str) -> Response:
        memory_services.delete_document(memory_id, doc_id, user=request.user)
        return Response(status=NO_CONTENT)

    @action(detail=True, methods=["get"], url_path=rf"documents/(?P<doc_id>{ID})/status")
    def document_status(self, request: Request, memory_id: str, doc_id: str) -> Response:
        status = memory_services.get_document_status(doc_id, user=request.user)
        return Response(s.DocumentStatusSerializer(status).data)

    @action(detail=True, methods=["post"], url_path=rf"threads/(?P<thread_id>{ID})")
    def link_thread(self, request: Request, memory_id: str, thread_id: str) -> Response:
        memory_services.link_memory_to_thread(memory_id, thread_id, user=request.user)
        return Response(status=NO_CONTENT)

    @link_thread.mapping.delete
    def unlink_thread(self, request: Request, memory_id: str, thread_id: str) -> Response:
        memory_services.unlink_memory_from_thread(memory_id, thread_id, user=request.user)
        return Response(status=NO_CONTENT)

    @action(detail=True, methods=["get"])
    def users(self, request: Request, memory_id: str) -> Response:
        limit, offset = self.page(request)
        users = memory_services.list_memory_users(
            memory_id, user=request.user, limit=limit, offset=offset
        )
        return Response(s.MemoryMemberSerializer(users, many=True).data)

    @users.mapping.post
    def add_user(self, request: Request, memory_id: str) -> Response:
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        entry = memory_services.add_memory_user(
            memory_id,
            serializer.validated_data["user_id"],
            serializer.validated_data["can_manage"],
            user=request.user,
        )
        return Response(s.MemoryMemberSerializer(entry).data, status=201)

    @action(detail=True, methods=["patch"], url_path=rf"users/(?P<user_id>{ID})")
    def update_user(self, request: Request, memory_id: str, user_id: str) -> Response:
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        entry = memory_services.update_memory_user(
            memory_id, user_id, serializer.validated_data["can_manage"], user=request.user
        )
        return Response(s.MemoryMemberSerializer(entry).data)

    @update_user.mapping.delete
    def remove_user(self, request: Request, memory_id: str, user_id: str) -> Response:
        memory_services.remove_memory_user(memory_id, user_id, user=request.user)
        return Response(status=NO_CONTENT)

    @action(detail=True, methods=["get"])
    def groups(self, request: Request, memory_id: str) -> Response:
        groups = memory_services.list_memory_groups(memory_id, user=request.user)
        return Response(s.MemoryGroupMemberSerializer(groups, many=True).data)

    @groups.mapping.post
    def add_group(self, request: Request, memory_id: str) -> Response:
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        entry = memory_services.add_memory_group(
            memory_id,
            serializer.validated_data["group_id"],
            serializer.validated_data["can_manage"],
            user=request.user,
        )
        return Response(s.MemoryGroupMemberSerializer(entry).data, status=201)

    @action(detail=True, methods=["delete"], url_path=r"groups/(?P<group_id>\d+)")
    def remove_group(self, request: Request, memory_id: str, group_id: str) -> Response:
        memory_services.remove_memory_group(memory_id, int(group_id), user=request.user)
        return Response(status=NO_CONTENT)

    def _with_permissions(self, request: Request, memory: Any) -> dict[str, Any]:
        data = {
            **memory.model_dump(),
            "permissions": object_permissions.memory_permissions(request.user, memory.id),
        }
        return s.MemoryWithPermissionsSerializer(data).data


class WorkflowViewSet(ApiViewSet):
    """Stored workflow definitions, their runs, and ad-hoc runs."""

    lookup_field = "workflow_id"
    lookup_value_regex = ID
    serializer_classes = {
        "create": s.WorkflowCreateSerializer,
        "partial_update": s.WorkflowUpdateSerializer,
        "run_definition": s.WorkflowRunRequestSerializer,
        "run": s.WorkflowRunByIdSerializer,
    }

    def list(self, request: Request) -> Response:
        limit, offset = self.page(request)
        records = workflow_services.list_workflows(user=request.user, limit=limit, offset=offset)
        return Response(s.WorkflowSerializer(records, many=True).data)

    def create(self, request: Request) -> Response:
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        record = workflow_services.create_workflow(
            serializer.validated_data["name"],
            serializer.validated_data["workflow"],
            user=request.user,
        )
        return Response(s.WorkflowSerializer(record).data, status=201)

    def retrieve(self, request: Request, workflow_id: str) -> Response:
        record = workflow_services.get_workflow(workflow_id, user=request.user)
        return Response(s.WorkflowSerializer(record).data)

    def partial_update(self, request: Request, workflow_id: str) -> Response:
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        record = workflow_services.update_workflow(
            workflow_id,
            user=request.user,
            name=data.get("name"),
            workflow=data.get("workflow"),
            active=data.get("active"),
        )
        return Response(s.WorkflowSerializer(record).data)

    def destroy(self, request: Request, workflow_id: str) -> Response:
        workflow_services.delete_workflow(workflow_id, user=request.user)
        return Response(status=NO_CONTENT)

    @action(detail=False, methods=["post"], url_path="run")
    def run_definition(self, request: Request) -> Response:
        """Run a definition sent in the request, without storing it."""
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        run = workflow_services.run_workflow(
            serializer.validated_data["workflow"],
            inputs=serializer.validated_data["inputs"],
            user=request.user,
        )
        return Response({"run_id": str(run.id), "status": run.status}, status=202)

    @action(detail=False, methods=["get"], url_path="actions")
    def list_actions(self, request: Request) -> Response:
        return Response(WorkflowService.list_actions())

    @action(detail=True, methods=["post"])
    def run(self, request: Request, workflow_id: str) -> Response:
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        run = workflow_services.run_workflow_by_id(
            workflow_id,
            inputs=serializer.validated_data["inputs"],
            user=request.user,
            run_id=serializer.validated_data["run_id"],
        )
        return Response({"run_id": str(run.id), "status": run.status}, status=202)

    @action(detail=True, methods=["get"])
    def runs(self, request: Request, workflow_id: str) -> Response:
        limit, offset = self.page(request)
        runs = workflow_services.list_workflow_runs(
            workflow_id, user=request.user, limit=limit, offset=offset
        )
        return Response(s.WorkflowRunSerializer(runs, many=True).data)

    @action(detail=True, methods=["get"], url_path=rf"runs/(?P<run_id>{ID})")
    def run_detail(self, request: Request, workflow_id: str, run_id: str) -> Response:
        run = workflow_services.get_workflow_run(run_id, user=request.user)
        if run is None:
            # Absent covers both "no such run" and "not yours", by design.
            raise NotFound("Run not found")
        return Response(s.WorkflowRunDetailSerializer(run).data)


class IntegrationViewSet(ApiViewSet):
    """Integrations the caller may use, with live status; connect/disconnect/reconnect.

    The OAuth callback is a plain Django view at a fixed URL:
    ``include("django_ai_sdk.integrations.mcp.urls")``.
    """

    lookup_field = "name"
    lookup_value_regex = ID

    def list(self, request: Request) -> Response:
        integrations = integration_services.list_integrations_for_user(request.user)
        return Response(s.IntegrationSerializer(integrations, many=True).data)

    @action(detail=True, methods=["post"])
    def connect(self, request: Request, name: str) -> Response:
        redirect_uri = request.build_absolute_uri(
            reverse("integrations_mcp:oauth-callback", kwargs={"server_name": name})
        )
        result = integration_services.connect_integration(
            name, request.user, request=request._request, redirect_uri=redirect_uri
        )
        if result is None:
            raise NotFound("Unknown integration")
        return Response({"redirect_url": result["redirect_url"]})

    @action(detail=True, methods=["post"])
    def disconnect(self, request: Request, name: str) -> Response:
        deleted = integration_services.disconnect_integration(name, request.user)
        if deleted is None:
            raise NotFound("Unknown integration")
        if not deleted:
            raise NotFound("Not connected")
        return Response(status=NO_CONTENT)

    @action(detail=True, methods=["post"])
    def reconnect(self, request: Request, name: str) -> Response:
        status = integration_services.reconnect_integration(name, request.user)
        if status is None:
            raise NotFound("Unknown integration")
        return Response({"status": status})
