from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from django.contrib.auth import get_user_model
from django.http import HttpRequest, StreamingHttpResponse
from django.urls import path
from django.views import View
from django_ai_sdk import Agent
from django_ai_sdk.agents.services import (
    AgentCreateData,
    AgentService,
    AgentUpdateData,
    add_agent_group,
    get_agent_info,
    list_agent_groups,
    list_agents,
    remove_agent_group,
)
from django_ai_sdk.common import ChatMessage
from django_ai_sdk.logger import get_logger
from django_ai_sdk.memories.services import link_memories, unlink_memories
from django_ai_sdk.permissions import Operation, PermissionDenied
from django_ai_sdk.protocols.utils import format_sse
from django_ai_sdk.storage.services import (
    create_thread,
    delete_all_threads,
    delete_message,
    delete_thread,
    get_thread,
    get_thread_file_meta,
    get_thread_history,
    list_threads,
    rate_message,
    restore_message,
    update_thread,
)
from django_ai_sdk.tracing.services import (
    message_token_usage,
    message_traces,
    thread_token_usage,
    thread_traces,
)
from django_ai_sdk.views.schemas import Message
from rest_framework import serializers
from rest_framework.exceptions import ValidationError
from rest_framework.response import Response
from rest_framework.views import APIView

if TYPE_CHECKING:
    from rest_framework.request import Request

logger = get_logger(__name__)


class ThreadListItemSerializer(serializers.Serializer):
    id = serializers.CharField()
    title = serializers.CharField()
    agent_id = serializers.CharField()
    created_at = serializers.CharField()
    updated_at = serializers.CharField()
    message_count = serializers.IntegerField()


class ThreadListResponseSerializer(serializers.Serializer):
    threads = ThreadListItemSerializer(many=True)


class FeedbackResponseSerializer(serializers.Serializer):
    id = serializers.CharField()
    user_id = serializers.CharField(allow_null=True)
    rating = serializers.IntegerField()
    feedback = serializers.CharField()
    created_at = serializers.CharField(allow_null=True)


class ThreadMessageSerializer(serializers.Serializer):
    id = serializers.CharField()
    role = serializers.CharField()
    parts = serializers.ListField(default=[])
    finish_reason = serializers.CharField(allow_null=True, required=False)
    tool_calls = serializers.ListField(default=[])
    processing_time_ms = serializers.IntegerField(allow_null=True, required=False)
    has_errors = serializers.BooleanField(default=False)
    usage = serializers.DictField(allow_null=True, required=False)
    feedback = FeedbackResponseSerializer(allow_null=True, required=False)
    created_at = serializers.CharField(allow_null=True, required=False)


class ThreadDetailSerializer(serializers.Serializer):
    thread = serializers.DictField()
    messages = ThreadMessageSerializer(many=True)


class ThreadFileMetaSerializer(serializers.Serializer):
    file_count = serializers.IntegerField()
    file_memory_id = serializers.CharField(allow_null=True)


class CreateThreadResponseSerializer(serializers.Serializer):
    thread_id = serializers.CharField(allow_null=True)


class MessageResponseSerializer(serializers.Serializer):
    id = serializers.CharField()
    is_deleted = serializers.BooleanField(allow_null=True)
    feedback = FeedbackResponseSerializer(allow_null=True, required=False)


class DeleteAllThreadsResponseSerializer(serializers.Serializer):
    success = serializers.BooleanField()
    deleted_count = serializers.IntegerField()


class AgentInfoSerializer(serializers.Serializer):
    id = serializers.CharField()
    name = serializers.CharField(allow_null=True)
    model = serializers.CharField(allow_null=True)
    class_name = serializers.CharField()
    description = serializers.CharField(allow_null=True, required=False)
    instructions = serializers.CharField(allow_null=True, required=False)
    file_upload = serializers.BooleanField(default=False)
    rag = serializers.BooleanField(default=False)


class ListAgentsItemSerializer(serializers.Serializer):
    id = serializers.CharField()
    name = serializers.CharField()
    model = serializers.CharField()


class ListAgentsSerializer(serializers.Serializer):
    agents = ListAgentsItemSerializer(many=True)


class ToolSerializer(serializers.Serializer):
    label = serializers.CharField()
    description = serializers.CharField(allow_null=True, required=False)


class IntegrationStatusSerializer(serializers.Serializer):
    server_name = serializers.CharField()
    label = serializers.CharField()
    type = serializers.CharField()
    status = serializers.CharField()
    tool_names = serializers.ListField(child=serializers.CharField())


class ToolsResponseSerializer(serializers.Serializer):
    tools = ToolSerializer(many=True)
    integrations = IntegrationStatusSerializer(many=True, default=[])


class ReindexResponseSerializer(serializers.Serializer):
    success = serializers.BooleanField()
    message = serializers.CharField()


class MessagePartSerializer(serializers.Serializer):
    type = serializers.CharField()
    text = serializers.CharField(required=False, allow_null=True)


class MessageSerializer(serializers.Serializer):
    role = serializers.CharField()
    parts = MessagePartSerializer(many=True)
    id = serializers.CharField(required=False, allow_null=True)


class ChatRequestSerializer(serializers.Serializer):
    messages = MessageSerializer(many=True)
    agent_id = serializers.CharField(required=False, default="")


class ThreadListAPIView(APIView):
    def get(self, request: Request) -> Response:
        limit = int(request.query_params.get("limit", 100))
        offset = int(request.query_params.get("offset", 0))
        threads = list_threads(user=request.user, limit=limit, offset=offset)
        items = [
            {
                "id": t.id,
                "title": t.title,
                "agent_id": t.agent_id,
                "created_at": t.created_at.isoformat(),
                "updated_at": t.updated_at.isoformat(),
                "message_count": t.message_count,
            }
            for t in threads
        ]
        return Response(ThreadListResponseSerializer({"threads": items}).data)


class ThreadCreateAPIView(APIView):
    def post(self, request: Request) -> Response:
        try:
            serializer = ChatRequestSerializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            agent_id = request.data.get("agent_id", "")  # type: ignore[union-attr]
            thread_id = create_thread(
                agent_id=agent_id,
                user=request.user,
                # Initial messages are not persisted here; the chat/stream endpoint
                # receives and stores the full message list.
            )
            return Response(CreateThreadResponseSerializer({"thread_id": thread_id}).data)
        except PermissionDenied as e:
            return Response({"message": str(e)}, status=403)
        except ValueError as e:
            return Response({"message": str(e)}, status=400)
        except Exception as e:
            return Response({"message": str(e)}, status=500)


class ThreadDetailAPIView(APIView):
    def get(self, request: Request, thread_id: str) -> Response:
        try:
            data = get_thread_history(thread_id, user=request.user)

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

            return Response(ThreadDetailSerializer(data).data)
        except PermissionDenied as e:
            return Response({"message": str(e)}, status=403)
        except ValueError as e:
            return Response({"message": str(e)}, status=404)

    def patch(self, request: Request, thread_id: str) -> Response:
        agent_id = request.data.get("agent_id")  # type: ignore[union-attr]
        if not agent_id:
            return Response({"message": "agent_id required"}, status=400)
        try:
            AgentService.from_registry(agent_id)
            thread = get_thread(thread_id, user=request.user)
            if thread is None:
                return Response({"message": "Thread not found"}, status=404)
            if thread.agent_id:
                unlink_memories(thread.agent_id, thread_id, user=request.user)
            update_thread(thread_id, metadata={"agent_id": agent_id}, user=request.user)
            link_memories(agent_id, thread_id, user=request.user)
            return Response({"success": True})
        except PermissionDenied as e:
            return Response({"message": str(e)}, status=403)
        except ValueError as e:
            return Response({"message": str(e)}, status=400)


class ThreadFileMetaAPIView(APIView):
    def get(self, request: Request, thread_id: str) -> Response:
        try:
            data = get_thread_file_meta(thread_id, user=request.user)
            return Response(ThreadFileMetaSerializer(data).data)
        except ValueError as e:
            return Response({"message": str(e)}, status=404)


class ThreadTracesAPIView(APIView):
    def get(self, request: Request, thread_id: str) -> Response:
        try:
            traces = thread_traces(
                thread_id,
                user=request.user,
                message_id=request.query_params.get("message_id"),
                operation_name=request.query_params.get("operation_name"),
                limit=int(request.query_params.get("limit", 100)),
                offset=int(request.query_params.get("offset", 0)),
            )
            return Response({"traces": [t.model_dump(mode="json") for t in traces]})
        except PermissionDenied as e:
            return Response({"message": str(e)}, status=403)
        except ValueError as e:
            return Response({"message": str(e)}, status=404)
        except Exception as e:
            return Response({"message": str(e)}, status=500)


class MessageTracesAPIView(APIView):
    def get(self, request: Request, message_id: str) -> Response:
        try:
            traces = message_traces(
                message_id,
                user=request.user,
                operation_name=request.query_params.get("operation_name"),
                limit=int(request.query_params.get("limit", 100)),
                offset=int(request.query_params.get("offset", 0)),
            )
            return Response({"traces": [t.model_dump(mode="json") for t in traces]})
        except PermissionDenied as e:
            return Response({"message": str(e)}, status=403)
        except ValueError as e:
            return Response({"message": str(e)}, status=404)
        except Exception as e:
            return Response({"message": str(e)}, status=500)


class ThreadTokenUsageAPIView(APIView):
    def get(self, request: Request, thread_id: str) -> Response:
        try:
            usage = thread_token_usage(thread_id, user=request.user)
            return Response(usage.model_dump())
        except PermissionDenied as e:
            return Response({"message": str(e)}, status=403)
        except ValueError as e:
            return Response({"message": str(e)}, status=404)
        except Exception as e:
            return Response({"message": str(e)}, status=500)


class MessageTokenUsageAPIView(APIView):
    def get(self, request: Request, message_id: str) -> Response:
        try:
            usage = message_token_usage(message_id, user=request.user)
            return Response(usage.model_dump())
        except PermissionDenied as e:
            return Response({"message": str(e)}, status=403)
        except ValueError as e:
            return Response({"message": str(e)}, status=404)
        except Exception as e:
            return Response({"message": str(e)}, status=500)


class ThreadDeleteAPIView(APIView):
    def delete(self, request: Request, thread_id: str) -> Response:
        try:
            success = delete_thread(thread_id, user=request.user)
            if success:
                return Response({"success": True, "message": "Thread deleted successfully"})
            return Response({"message": "Thread not found"}, status=404)
        except PermissionDenied as e:
            return Response({"message": str(e)}, status=403)
        except ValueError as e:
            return Response({"message": str(e)}, status=404)


class ThreadDeleteAllAPIView(APIView):
    def delete(self, request: Request) -> Response:
        try:
            deleted_count = delete_all_threads(user=request.user)
            return Response(
                DeleteAllThreadsResponseSerializer(
                    {"success": True, "deleted_count": deleted_count}
                ).data
            )
        except PermissionDenied as e:
            return Response({"message": str(e)}, status=403)
        except Exception as e:
            return Response({"message": str(e)}, status=500)


class RateMessageAPIView(APIView):
    def post(self, request: Request, thread_id: str, message_id: str) -> Response:
        if "rating" not in request.data:  # type: ignore[operator]
            return Response({"message": "rating is required"}, status=400)
        rating = request.data.get("rating")  # type: ignore[union-attr]
        feedback_text = request.data.get("feedback", "")  # type: ignore[union-attr]
        try:
            rate_message(thread_id, message_id, rating, feedback=feedback_text, user=request.user)  # type: ignore[arg-type]
            return Response(MessageResponseSerializer({"id": message_id, "is_deleted": False}).data)
        except PermissionDenied as e:
            return Response({"message": str(e)}, status=403)
        except ValueError as e:
            return Response({"message": str(e)}, status=404)


class DeleteMessageAPIView(APIView):
    def post(self, request: Request, thread_id: str, message_id: str) -> Response:
        try:
            delete_message(thread_id, message_id, user=request.user)
            return Response(MessageResponseSerializer({"id": message_id, "is_deleted": True}).data)
        except PermissionDenied as e:
            return Response({"message": str(e)}, status=403)
        except ValueError as e:
            return Response({"message": str(e)}, status=404)


class RestoreMessageAPIView(APIView):
    def post(self, request: Request, thread_id: str, message_id: str) -> Response:
        try:
            restore_message(thread_id, message_id, user=request.user)
            return Response(MessageResponseSerializer({"id": message_id, "is_deleted": False}).data)
        except PermissionDenied as e:
            return Response({"message": str(e)}, status=403)
        except ValueError as e:
            return Response({"message": str(e)}, status=404)


class ListAgentsAPIView(APIView):
    def get(self, request: Request) -> Response:
        limit = int(request.query_params.get("limit", 100))
        offset = int(request.query_params.get("offset", 0))
        try:
            items = list_agents(user=request.user, limit=limit, offset=offset)
            return Response(ListAgentsSerializer({"agents": items}).data)
        except PermissionDenied as e:
            return Response({"message": str(e)}, status=403)


class AgentInfoAPIView(APIView):
    def get(self, request: Request, agent_id: str) -> Response:
        try:
            agent = AgentService.from_registry(agent_id)
            info = get_agent_info(agent_id, user=request.user)
            return Response(
                AgentInfoSerializer(
                    {
                        "id": info.id,
                        "name": info.name,
                        "model": info.model,
                        "class_name": info.class_name,
                        "description": info.description,
                        "instructions": agent.get_system_prompt(),
                        "file_upload": info.file_upload,
                        "rag": info.rag,
                    }
                ).data
            )
        except PermissionDenied as e:
            return Response({"message": str(e)}, status=403)
        except ValueError as e:
            return Response({"message": str(e)}, status=404)


class AgentToolsAPIView(APIView):
    async def get(self, request: Request, agent_id: str) -> Response:
        try:
            agent = await AgentService.get(agent_id)
            await AgentService.has_perms(
                request.user,
                Operation.VIEW_AGENT,
                obj=agent.config if agent.is_runtime else None,
                agent=agent,
            )
        except ValueError as e:
            return Response({"message": str(e)}, status=404)
        except PermissionDenied as e:
            return Response({"message": str(e)}, status=403)

        tools_data = []
        try:
            tool_objs = await agent.get_tools()
            tools_data = [
                {
                    "label": getattr(t, "label", None) or t.name.replace("_", " ").title(),
                    "description": t.description or "",
                }
                for t in tool_objs
            ]
        except Exception:
            logger.exception("Failed to build tools for agent %s", agent_id)

        integrations_data = []
        try:
            integration_status = await AgentService.get_integration_status(agent, user=request.user)
            integrations_data = [
                {
                    "server_name": s.server_name,
                    "label": s.label,
                    "type": s.type,
                    "status": s.status,
                    "tool_names": s.tool_names,
                }
                for s in integration_status
            ]
        except Exception:
            logger.exception("Failed to load integration status for agent %s", agent_id)

        return Response(
            ToolsResponseSerializer({"tools": tools_data, "integrations": integrations_data}).data
        )


class ReindexAgentAPIView(APIView):
    def post(self, request: Request, agent_id: str) -> Response:
        memory_id = request.data.get("memory_id")  # type: ignore[union-attr]
        force_rebuild = request.data.get("force_rebuild", False)  # type: ignore[union-attr]
        try:
            agent = AgentService.from_registry(agent_id)
            result = Agent.reindex(agent, memory_id, force_rebuild)  # type: ignore[arg-type]

            if not result:
                return Response(
                    ReindexResponseSerializer(
                        {
                            "success": False,
                            "message": "No RAG provider configured for this agent",
                        }
                    ).data
                )

            rebuild_msg = " (force rebuild)" if force_rebuild else ""
            message = "RAG pipeline reindexed successfully" + rebuild_msg
            if memory_id:
                message += f" for memory {memory_id}"

            return Response(ReindexResponseSerializer({"success": True, "message": message}).data)
        except ValueError as e:
            return Response({"message": str(e)}, status=404)


class AgentStatelessRunAPIView(APIView):
    """Stateless run"""

    async def post(self, request: Request, agent_id: str) -> Response:
        try:
            serializer = ChatRequestSerializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            messages = [Message(**m) for m in serializer.validated_data["messages"]]  # type: ignore[index, optional-subscript]
            agent = await AgentService.get(agent_id)
            chat_messages = agent.protocol_handler.to_chat_messages(messages)
            result = await agent.run(chat_messages, user=request.user)
            return Response({"result": result, "thread_id": None})
        except PermissionDenied as e:
            return Response({"message": str(e)}, status=403)
        except ValidationError as e:
            return Response({"message": str(e)}, status=400)
        except ValueError as e:
            return Response({"message": str(e)}, status=404)
        except NotImplementedError as e:
            return Response({"message": str(e)}, status=501)


class AgentRunAPIView(APIView):
    """Synchronous JSON endpoint wrapping Agent.run()"""

    async def post(self, request: Request, thread_id: str) -> Response:
        try:
            serializer = ChatRequestSerializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            messages = [Message(**m) for m in serializer.validated_data["messages"]]  # type: ignore[index, optional-subscript]
            agent = await AgentService.get_agent(thread_id, user=request.user)
            chat_messages = agent.protocol_handler.to_chat_messages(messages)
            result = await agent.run(chat_messages, thread_id=thread_id, user=request.user)
            return Response({"result": result, "thread_id": thread_id})
        except PermissionDenied as e:
            return Response({"message": str(e)}, status=403)
        except ValidationError as e:
            return Response({"message": str(e)}, status=400)
        except ValueError as e:
            return Response({"message": str(e)}, status=404)
        except NotImplementedError as e:
            return Response({"message": str(e)}, status=501)
        except Exception as e:
            return Response({"message": str(e)}, status=500)


class AgentAPIView(View):
    async def post(self, request: HttpRequest, thread_id: str) -> StreamingHttpResponse:
        try:
            serializer = ChatRequestSerializer(data=json.loads(request.body))
            serializer.is_valid(raise_exception=True)
            messages = [Message(**m) for m in serializer.validated_data["messages"]]  # type: ignore[index, optional-subscript]
            agent = await AgentService.get_agent(thread_id, user=request.user)
            return await agent.as_view(messages, thread_id=thread_id, user=request.user)
        except PermissionDenied as e:
            return self._error_response({"message": str(e)}, 403)
        except ValidationError as e:
            return self._error_response({"message": str(e)}, 400)
        except ValueError as e:
            return self._error_response({"message": str(e)}, 404)
        except Exception as e:
            return self._error_response({"message": str(e)}, 500)

    @staticmethod
    def _error_response(data: dict[str, Any], status: int) -> StreamingHttpResponse:
        return StreamingHttpResponse(
            content=format_sse(data),
            content_type="text/event-stream",
            status=status,
        )


# ============================================================================
# Runtime Agents (DB-configured)
# ============================================================================


class AgentUserInSerializer(serializers.Serializer):
    user_id = serializers.CharField()
    can_manage = serializers.BooleanField(default=False)


class AgentGroupInSerializer(serializers.Serializer):
    group_id = serializers.IntegerField()
    can_manage = serializers.BooleanField(default=False)


class AgentSettingsSerializer(serializers.Serializer):
    id = serializers.UUIDField(read_only=True)
    name = serializers.CharField()
    slug = serializers.SlugField()
    agent = serializers.CharField(allow_blank=True, default="")
    model = serializers.CharField()
    system_prompt = serializers.CharField(allow_blank=True)
    tools = serializers.ListField(child=serializers.CharField(), default=list)
    integrations = serializers.ListField(child=serializers.CharField(), default=list)
    suggestion_enabled = serializers.BooleanField(default=False)
    title_generation = serializers.BooleanField(default=True)
    max_history = serializers.IntegerField(allow_null=True, required=False)
    file_upload = serializers.BooleanField(default=False)
    active = serializers.BooleanField(default=True)
    created_at = serializers.DateTimeField(read_only=True)
    updated_at = serializers.DateTimeField(read_only=True)


class AgentSettingsCreateSerializer(serializers.Serializer):
    name = serializers.CharField()
    slug = serializers.SlugField(default="")
    agent = serializers.CharField(allow_blank=True, default="")
    model = serializers.CharField(default="gpt-4o")
    system_prompt = serializers.CharField(allow_blank=True, default="")
    tools = serializers.ListField(child=serializers.CharField(), default=list)
    integrations = serializers.ListField(child=serializers.CharField(), default=list)
    users = AgentUserInSerializer(many=True, required=False)
    groups = AgentGroupInSerializer(many=True, required=False)
    suggestion_enabled = serializers.BooleanField(default=False)
    title_generation = serializers.BooleanField(default=True)
    max_history = serializers.IntegerField(allow_null=True, required=False)
    file_upload = serializers.BooleanField(default=False)
    users = AgentUserInSerializer(many=True, required=False, default=list)


class AgentSettingsUpdateSerializer(serializers.Serializer):
    name = serializers.CharField(required=False)
    agent = serializers.CharField(allow_blank=True, required=False)
    model = serializers.CharField(required=False)
    system_prompt = serializers.CharField(allow_blank=True, required=False)
    tools = serializers.ListField(child=serializers.CharField(), required=False)
    integrations = serializers.ListField(child=serializers.CharField(), required=False)
    users = AgentUserInSerializer(many=True, required=False)
    groups = AgentGroupInSerializer(many=True, required=False)
    suggestion_enabled = serializers.BooleanField(required=False)
    title_generation = serializers.BooleanField(required=False)
    max_history = serializers.IntegerField(allow_null=True, required=False)
    file_upload = serializers.BooleanField(required=False)
    active = serializers.BooleanField(required=False)
    users = AgentUserInSerializer(many=True, required=False)


class RuntimeAgentBasesAPIView(APIView):
    def get(self, request: Request) -> Response:
        from django_ai_sdk.agents.config import get_runtime_agent_bases

        data = [
            {"path": f"{cls.__module__}.{cls.__qualname__}", "name": cls.__name__}
            for cls in get_runtime_agent_bases()
        ]
        return Response(data)


class RuntimeAgentToolsAPIView(APIView):
    def get(self, request: Request) -> Response:
        from django_ai_sdk.agents.config import get_tool_registry

        data = [{"key": k, "path": v} for k, v in get_tool_registry().items()]
        return Response(data)


class RuntimeAgentListCreateAPIView(APIView):
    async def get(self, request: Request) -> Response:
        configs = await AgentService.list_runtime_agents(user=request.user)
        return Response(AgentSettingsSerializer(configs, many=True).data)

    async def post(self, request: Request) -> Response:
        serializer = AgentSettingsCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        skip_keys = {"users", "groups"}
        data = {k: v for k, v in serializer.validated_data.items() if k not in skip_keys}  # type: ignore[union-attr]
        try:
            config = await AgentService.create_runtime_agent(
                cast("AgentCreateData", data),
                user=request.user,
            )
        except Exception as e:
            return Response({"message": str(e)}, status=400)
        for entry in serializer.validated_data.get("users") or []:  # type: ignore[union-attr]
            try:
                await AgentService.add_agent_user(
                    str(config.id),
                    entry["user_id"],
                    entry.get("can_manage", False),
                    user=request.user,
                )
            except Exception:
                pass
        for entry in serializer.validated_data.get("groups") or []:  # type: ignore[union-attr]
            try:
                await AgentService.add_agent_group(
                    str(config.id),
                    entry["group_id"],
                    entry.get("can_manage", False),
                    user=request.user,
                )
            except Exception:
                pass
        return Response(AgentSettingsSerializer(config).data)


class RuntimeAgentDetailAPIView(APIView):
    async def get(self, request: Request, runtime_id: str) -> Response:
        try:
            config = await AgentService.get_runtime_agent(runtime_id, user=request.user)
            return Response(AgentSettingsSerializer(config).data)
        except ValueError as e:
            return Response({"message": str(e)}, status=404)

    async def patch(self, request: Request, runtime_id: str) -> Response:
        serializer = AgentSettingsUpdateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        skip_keys = {"users", "groups"}
        data = {k: v for k, v in serializer.validated_data.items() if k not in skip_keys}  # type: ignore[union-attr]
        try:
            config = await AgentService.update_runtime_agent(
                runtime_id,
                cast("AgentUpdateData", data),
                user=request.user,
            )
        except ValueError as e:
            return Response({"message": str(e)}, status=404)
        except Exception as e:
            return Response({"message": str(e)}, status=400)
        for entry in serializer.validated_data.get("users") or []:  # type: ignore[union-attr]
            try:
                await AgentService.add_agent_user(
                    runtime_id, entry["user_id"], entry.get("can_manage", False), user=request.user
                )
            except Exception:
                pass
        for entry in serializer.validated_data.get("groups") or []:  # type: ignore[union-attr]
            try:
                await AgentService.add_agent_group(
                    runtime_id, entry["group_id"], entry.get("can_manage", False), user=request.user
                )
            except Exception:
                pass
        return Response(AgentSettingsSerializer(config).data)

    async def delete(self, request: Request, runtime_id: str) -> Response:
        try:
            config = await AgentService.delete_runtime_agent(runtime_id, user=request.user)
            return Response(AgentSettingsSerializer(config).data)
        except ValueError as e:
            return Response({"message": str(e)}, status=404)


# ── Agent Users ───────────────────────────────────────────────────────────


class AgentUserOutSerializer(serializers.Serializer):
    user_id = serializers.CharField()
    can_manage = serializers.BooleanField()
    created_at = serializers.CharField()


class AddAgentUserInSerializer(serializers.Serializer):
    user_id = serializers.CharField()
    can_manage = serializers.BooleanField(default=False)


# ── Agent Groups ──────────────────────────────────────────────────────────


class AgentGroupOutSerializer(serializers.Serializer):
    group_id = serializers.IntegerField()
    group_name = serializers.CharField(source="group.name")
    can_manage = serializers.BooleanField()
    created_at = serializers.CharField()


class AddAgentGroupInSerializer(serializers.Serializer):
    group_id = serializers.IntegerField()
    can_manage = serializers.BooleanField(default=False)


class AgentGroupListCreateAPIView(APIView):
    def get(self, request: Request, runtime_id: str) -> Response:
        try:
            groups = list_agent_groups(runtime_id, user=request.user)
        except PermissionDenied as e:
            return Response({"detail": str(e)}, status=403)
        except ValueError as e:
            return Response({"detail": str(e)}, status=404)
        return Response(AgentGroupOutSerializer(groups, many=True).data)

    def post(self, request: Request, runtime_id: str) -> Response:
        serializer = AddAgentGroupInSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            entry = add_agent_group(
                runtime_id,
                serializer.validated_data["group_id"],  # type: ignore[index]
                serializer.validated_data.get("can_manage", False),  # type: ignore[union-attr]
                user=request.user,
            )
        except PermissionDenied as e:
            return Response({"detail": str(e)}, status=403)
        except ValueError as e:
            return Response({"detail": str(e)}, status=404)
        return Response(AgentGroupOutSerializer(entry).data)


class AgentGroupDetailAPIView(APIView):
    def delete(self, request: Request, runtime_id: str, group_id: int) -> Response:
        try:
            remove_agent_group(runtime_id, group_id, user=request.user)
        except PermissionDenied as e:
            return Response({"detail": str(e)}, status=403)
        except ValueError as e:
            return Response({"detail": str(e)}, status=404)
        return Response(status=204)


# ============================================================================
# Workflows
# ============================================================================


class WorkflowSerializer(serializers.Serializer):
    id = serializers.UUIDField(read_only=True)
    name = serializers.CharField()
    definition = serializers.DictField()
    active = serializers.BooleanField(default=True)
    created_at = serializers.DateTimeField(read_only=True)
    updated_at = serializers.DateTimeField(read_only=True)


class WorkflowCreateSerializer(serializers.Serializer):
    name = serializers.CharField()
    workflow = serializers.DictField()


class WorkflowUpdateSerializer(serializers.Serializer):
    name = serializers.CharField(required=False)
    workflow = serializers.DictField(required=False)
    active = serializers.BooleanField(required=False)


class WorkflowListCreateAPIView(APIView):
    async def get(self, request: Request) -> Response:
        from django_ai_sdk.workflows import WorkflowService

        limit = int(request.query_params.get("limit", 100))
        offset = int(request.query_params.get("offset", 0))
        records = await WorkflowService.list_workflows(limit=limit, offset=offset)
        return Response(WorkflowSerializer(records, many=True).data)

    async def post(self, request: Request) -> Response:
        from django_ai_sdk.workflows import WorkflowDefinition, WorkflowService

        serializer = WorkflowCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            workflow = WorkflowDefinition.model_validate(serializer.validated_data["workflow"])
            record = await WorkflowService.create(
                serializer.validated_data["name"], workflow, user=request.user
            )
            return Response(WorkflowSerializer(record).data, status=201)
        except Exception as e:
            return Response({"message": str(e)}, status=400)


class WorkflowDetailAPIView(APIView):
    async def get(self, request: Request, workflow_id: str) -> Response:
        from django_ai_sdk.workflows import WorkflowService
        from django_ai_sdk.workflows.models import WorkflowSettings

        try:
            record = await WorkflowService.get(workflow_id)
            return Response(WorkflowSerializer(record).data)
        except WorkflowSettings.DoesNotExist:
            return Response({"message": "Workflow not found"}, status=404)

    async def patch(self, request: Request, workflow_id: str) -> Response:
        from django_ai_sdk.workflows import WorkflowDefinition, WorkflowService
        from django_ai_sdk.workflows.models import WorkflowSettings

        serializer = WorkflowUpdateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            workflow_data = serializer.validated_data.get("workflow")
            workflow = WorkflowDefinition.model_validate(workflow_data) if workflow_data else None
            record = await WorkflowService.update(
                workflow_id,
                name=serializer.validated_data.get("name"),
                workflow=workflow,
                active=serializer.validated_data.get("active"),
            )
            return Response(WorkflowSerializer(record).data)
        except WorkflowSettings.DoesNotExist:
            return Response({"message": "Workflow not found"}, status=404)
        except Exception as e:
            return Response({"message": str(e)}, status=400)

    async def delete(self, request: Request, workflow_id: str) -> Response:
        from django_ai_sdk.workflows import WorkflowService
        from django_ai_sdk.workflows.models import WorkflowSettings

        try:
            await WorkflowService.get(workflow_id)
            await WorkflowService.delete(workflow_id)
            return Response(status=204)
        except WorkflowSettings.DoesNotExist:
            return Response({"message": "Workflow not found"}, status=404)


class WorkflowRunStepSerializer(serializers.Serializer):
    id = serializers.UUIDField()
    sequence = serializers.IntegerField()
    step_name = serializers.CharField()
    output_key = serializers.CharField()
    output = serializers.DictField(allow_null=True, required=False)
    status = serializers.CharField()
    error = serializers.CharField()
    started_at = serializers.DateTimeField(allow_null=True)
    completed_at = serializers.DateTimeField(allow_null=True)


class WorkflowRunSerializer(serializers.Serializer):
    id = serializers.UUIDField()
    workflow_id = serializers.UUIDField(allow_null=True)
    status = serializers.CharField()
    outputs = serializers.DictField(allow_null=True, required=False)
    error = serializers.CharField()
    created_at = serializers.DateTimeField()
    started_at = serializers.DateTimeField(allow_null=True)
    completed_at = serializers.DateTimeField(allow_null=True)


class WorkflowRunDetailSerializer(WorkflowRunSerializer):
    steps = WorkflowRunStepSerializer(many=True, source="steps.all")


class WorkflowRunAPIView(APIView):
    async def post(self, request: Request) -> Response:
        try:
            from django_ai_sdk.workflows import WorkflowDefinition, WorkflowService

            workflow = WorkflowDefinition.model_validate(request.data.get("workflow", {}))
            run = await WorkflowService.run(
                workflow,
                [ChatMessage(**m) for m in request.data.get("messages", [])],
                user=request.user,
            )
            return Response({"run_id": str(run.id), "status": run.status}, status=202)
        except Exception as e:
            return Response({"message": str(e)}, status=500)


class WorkflowRunByIdAPIView(APIView):
    async def post(self, request: Request, workflow_id: str) -> Response:
        try:
            from django_ai_sdk.workflows import WorkflowService
            from django_ai_sdk.workflows.models import WorkflowSettings

            run_id = request.data.get("run_id")
            run = await WorkflowService.run_by_id(
                workflow_id,
                [ChatMessage(**m) for m in request.data.get("messages", [])],
                user=request.user,
                run_id=run_id,
            )
            return Response({"run_id": str(run.id), "status": run.status}, status=202)
        except WorkflowSettings.DoesNotExist:
            return Response({"message": "Workflow not found"}, status=404)
        except Exception as e:
            return Response({"message": str(e)}, status=500)


class WorkflowRunListAPIView(APIView):
    async def get(self, request: Request, workflow_id: str) -> Response:
        from django_ai_sdk.workflows import WorkflowService

        limit = int(request.query_params.get("limit", 50))
        offset = int(request.query_params.get("offset", 0))
        try:
            runs = await WorkflowService.list_runs(workflow_id, limit=limit, offset=offset)
            return Response(WorkflowRunSerializer(runs, many=True).data)
        except Exception as e:
            return Response({"message": str(e)}, status=500)


class WorkflowRunDetailAPIView(APIView):
    async def get(self, request: Request, workflow_id: str, run_id: str) -> Response:
        from django_ai_sdk.workflows import WorkflowService
        from django_ai_sdk.workflows.models import WorkflowRun

        try:
            run = await WorkflowService.get_run(run_id)
            return Response(WorkflowRunDetailSerializer(run).data)
        except WorkflowRun.DoesNotExist:
            return Response({"message": "Run not found"}, status=404)
        except Exception as e:
            return Response({"message": str(e)}, status=500)


class WorkflowActionsAPIView(APIView):
    def get(self, request: Request) -> Response:
        from django_ai_sdk.workflows import WorkflowService

        return Response(WorkflowService.list_actions())


# ── Users ─────────────────────────────────────────────────────────────────────


class UserSerializer(serializers.Serializer):
    id = serializers.CharField()
    first_name = serializers.CharField()
    last_name = serializers.CharField()


class UserListAPIView(APIView):
    def get(self, request: Request) -> Response:
        User = get_user_model()
        qs = User.objects.order_by("first_name", "last_name")
        q = request.query_params.get("q", "").strip()
        if q:
            from django.db.models import Q

            qs = qs.filter(
                Q(first_name__icontains=q)
                | Q(last_name__icontains=q)
                | Q(email__icontains=q)
                | Q(username__icontains=q)
            )
        limit = min(int(request.query_params.get("limit", 10)), 100)
        serializer = UserSerializer(qs.values("id", "first_name", "last_name")[:limit], many=True)
        return Response(serializer.data)


class UserDetailSerializer(serializers.Serializer):
    id = serializers.CharField()
    first_name = serializers.CharField()
    last_name = serializers.CharField()
    email = serializers.EmailField()


class UserUpdateSerializer(serializers.Serializer):
    first_name = serializers.CharField(required=False)
    last_name = serializers.CharField(required=False)


class GroupOutSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    name = serializers.CharField()


class GroupSearchAPIView(APIView):
    def get(self, request: Request) -> Response:
        from django.contrib.auth.models import Group

        qs = Group.objects.order_by("name")
        q = request.query_params.get("q", "").strip()
        if q:
            qs = qs.filter(name__icontains=q)
        limit = min(int(request.query_params.get("limit", 10)), 100)
        serializer = GroupOutSerializer(qs.values("id", "name")[:limit], many=True)
        return Response(serializer.data)


class UserSessionSerializer(serializers.Serializer):
    session_key = serializers.CharField()
    ip = serializers.CharField()
    user_agent = serializers.CharField()
    created_at = serializers.DateTimeField()
    last_seen_at = serializers.DateTimeField()


class UserDetailAPIView(APIView):
    def get(self, request: Request, user_id: str) -> Response:
        User = get_user_model()
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response({"error": "User not found"}, status=404)
        return Response(UserDetailSerializer(user).data)

    def patch(self, request: Request, user_id: str) -> Response:
        User = get_user_model()
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response({"error": "User not found"}, status=404)
        serializer = UserUpdateSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)
        update_fields = []
        for field, value in serializer.validated_data.items():
            setattr(user, field, value)
            update_fields.append(field)
        if update_fields:
            user.save(update_fields=update_fields)
        return Response(UserDetailSerializer(user).data)


class UserSessionListAPIView(APIView):
    def get(self, request: Request, user_id: str) -> Response:
        from allauth.usersessions.models import UserSession

        sessions = UserSession.objects.filter(user_id=user_id).order_by("-last_seen_at")
        return Response(UserSessionSerializer(sessions, many=True).data)


# ── Agent Users ───────────────────────────────────────────────────────────


class AgentUserSerializer(serializers.Serializer):
    user_id = serializers.CharField(source="user.id")
    email = serializers.CharField(source="user.email")
    first_name = serializers.CharField(source="user.first_name")
    last_name = serializers.CharField(source="user.last_name")
    can_manage = serializers.BooleanField()
    created_at = serializers.DateTimeField()


class AgentUserAddSerializer(serializers.Serializer):
    user_id = serializers.CharField()
    can_manage = serializers.BooleanField(default=False)


class AgentUserUpdateSerializer(serializers.Serializer):
    can_manage = serializers.BooleanField()


class AgentUserListCreateAPIView(APIView):
    async def get(self, request: Request, runtime_id: str) -> Response:
        try:
            users = await AgentService.list_agent_users(runtime_id, user=request.user)
            return Response(AgentUserSerializer(users, many=True).data)
        except ValueError as e:
            return Response({"error": str(e)}, status=404)

    async def post(self, request: Request, runtime_id: str) -> Response:
        serializer = AgentUserAddSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)
        try:
            entry = await AgentService.add_agent_user(
                runtime_id,
                serializer.validated_data["user_id"],
                serializer.validated_data["can_manage"],
                user=request.user,
            )
            return Response(AgentUserSerializer(entry).data, status=201)
        except PermissionDenied as e:
            return Response({"error": str(e)}, status=403)
        except ValueError as e:
            return Response({"error": str(e)}, status=404)


class AgentUserDetailAPIView(APIView):
    async def patch(self, request: Request, runtime_id: str, user_id: str) -> Response:
        serializer = AgentUserUpdateSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)
        try:
            entry = await AgentService.update_agent_user(
                runtime_id,
                user_id,
                serializer.validated_data["can_manage"],
                user=request.user,
            )
            return Response(AgentUserSerializer(entry).data)
        except PermissionDenied as e:
            return Response({"error": str(e)}, status=403)
        except ValueError as e:
            return Response({"error": str(e)}, status=404)

    async def delete(self, request: Request, runtime_id: str, user_id: str) -> Response:
        try:
            await AgentService.remove_agent_user(runtime_id, user_id, user=request.user)
            return Response(status=204)
        except PermissionDenied as e:
            return Response({"error": str(e)}, status=403)
        except ValueError as e:
            return Response({"error": str(e)}, status=404)


urlpatterns = [
    path("agents/", ListAgentsAPIView.as_view(), name="agent-list"),
    path(
        "agents/<str:agent_id>/",
        AgentInfoAPIView.as_view(),
        name="agent-info",
    ),
    path(
        "agents/<str:agent_id>/tools/",
        AgentToolsAPIView.as_view(),
        name="agent-tools",
    ),
    path(
        "agents/<str:agent_id>/reindex/",
        ReindexAgentAPIView.as_view(),
        name="agent-reindex",
    ),
    path(
        "agents/<str:agent_id>/run/",
        AgentStatelessRunAPIView.as_view(),
        name="agent-run",
    ),
    path("threads/", ThreadListAPIView.as_view(), name="thread-list"),
    path("threads/<str:thread_id>/", ThreadDetailAPIView.as_view(), name="thread-detail"),
    path(
        "threads/<str:thread_id>/file-meta/",
        ThreadFileMetaAPIView.as_view(),
        name="thread-file-meta",
    ),
    path(
        "threads/<str:thread_id>/traces/",
        ThreadTracesAPIView.as_view(),
        name="thread-traces",
    ),
    path(
        "threads/<str:thread_id>/tokens/",
        ThreadTokenUsageAPIView.as_view(),
        name="thread-tokens",
    ),
    path(
        "messages/<str:message_id>/traces/",
        MessageTracesAPIView.as_view(),
        name="message-traces",
    ),
    path(
        "messages/<str:message_id>/tokens/",
        MessageTokenUsageAPIView.as_view(),
        name="message-tokens",
    ),
    path("threads/", ThreadCreateAPIView.as_view(), name="thread-create"),
    path("threads/<str:thread_id>/delete/", ThreadDeleteAPIView.as_view(), name="thread-delete"),
    path(
        "threads/<str:thread_id>/message/",
        AgentAPIView.as_view(),
        name="thread-message",
    ),
    path(
        "threads/<str:thread_id>/run/",
        AgentRunAPIView.as_view(),
        name="thread-run",
    ),
    path(
        "threads/delete-all/",
        ThreadDeleteAllAPIView.as_view(),
        name="thread-delete-all",
    ),
    path(
        "threads/<str:thread_id>/messages/<str:message_id>/rate/",
        RateMessageAPIView.as_view(),
        name="message-rate",
    ),
    path(
        "threads/<str:thread_id>/messages/<str:message_id>/delete/",
        DeleteMessageAPIView.as_view(),
        name="message-delete",
    ),
    path(
        "threads/<str:thread_id>/messages/<str:message_id>/restore/",
        RestoreMessageAPIView.as_view(),
        name="message-restore",
    ),
    path(
        "agents/runtimes/bases/",
        RuntimeAgentBasesAPIView.as_view(),
        name="runtime-agent-bases",
    ),
    path(
        "agents/runtimes/tools/",
        RuntimeAgentToolsAPIView.as_view(),
        name="runtime-agent-tools",
    ),
    path(
        "agents/runtimes/",
        RuntimeAgentListCreateAPIView.as_view(),
        name="runtime-agent-list",
    ),
    path(
        "agents/runtimes/<str:runtime_id>/",
        RuntimeAgentDetailAPIView.as_view(),
        name="runtime-agent-detail",
    ),
    path(
        "agents/runtimes/<str:runtime_id>/users/",
        AgentUserListCreateAPIView.as_view(),
        name="runtime-agent-user-list",
    ),
    path(
        "agents/runtimes/<str:runtime_id>/users/<str:user_id>/",
        AgentUserDetailAPIView.as_view(),
        name="runtime-agent-user-detail",
    ),
    path(
        "agents/runtimes/<str:runtime_id>/groups/",
        AgentGroupListCreateAPIView.as_view(),
        name="runtime-agent-group-list",
    ),
    path(
        "agents/runtimes/<str:runtime_id>/groups/<int:group_id>/",
        AgentGroupDetailAPIView.as_view(),
        name="runtime-agent-group-detail",
    ),
    path("workflows/", WorkflowListCreateAPIView.as_view(), name="workflow-list"),
    path("workflows/run/", WorkflowRunAPIView.as_view(), name="workflow-run"),
    path("workflows/actions/", WorkflowActionsAPIView.as_view(), name="workflow-actions"),
    path(
        "workflows/<str:workflow_id>/runs/",
        WorkflowRunListAPIView.as_view(),
        name="workflow-run-list",
    ),
    path(
        "workflows/<str:workflow_id>/runs/<str:run_id>/",
        WorkflowRunDetailAPIView.as_view(),
        name="workflow-run-detail",
    ),
    path("workflows/<str:workflow_id>/", WorkflowDetailAPIView.as_view(), name="workflow-detail"),
    path(
        "workflows/<str:workflow_id>/run/",
        WorkflowRunByIdAPIView.as_view(),
        name="workflow-run-by-id",
    ),
    path("users/", UserListAPIView.as_view(), name="user-list"),
    path("users/<str:user_id>/", UserDetailAPIView.as_view(), name="user-detail"),
    path("users/<str:user_id>/sessions/", UserSessionListAPIView.as_view(), name="user-sessions"),
    path("accounts/groups/", GroupSearchAPIView.as_view(), name="group-search"),
]
