import json
from typing import Any

from django.http import HttpRequest, StreamingHttpResponse
from django.urls import path
from django.views import View
from django_ai_sdk import Assistant
from django_ai_sdk.assistants.services import (
    AssistantService,
    AssistantSettingsService,
    add_assistant_group,
    add_assistant_user,
    get_assistant_info,
    list_assistant_groups,
    list_assistant_users,
    list_assistants,
    remove_assistant_group,
    remove_assistant_user,
    update_assistant_user,
)
from django_ai_sdk.logger import get_logger
from django_ai_sdk.memories.services import link_memories, unlink_memories
from django_ai_sdk.permissions import PermissionDenied
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
from django_ai_sdk.views.schemas import Message
from rest_framework import serializers
from rest_framework.exceptions import ValidationError
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

logger = get_logger(__name__)


class ThreadListItemSerializer(serializers.Serializer):
    id = serializers.CharField()
    title = serializers.CharField()
    assistant_id = serializers.CharField()
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


class AssistantInfoSerializer(serializers.Serializer):
    id = serializers.CharField()
    name = serializers.CharField(allow_null=True)
    model = serializers.CharField(allow_null=True)
    class_name = serializers.CharField()
    description = serializers.CharField(allow_null=True, required=False)
    instructions = serializers.CharField(allow_null=True, required=False)
    file_upload = serializers.BooleanField(default=False)


class ListAssistantsItemSerializer(serializers.Serializer):
    id = serializers.CharField()
    name = serializers.CharField()
    model = serializers.CharField()


class ListAssistantsSerializer(serializers.Serializer):
    assistants = ListAssistantsItemSerializer(many=True)


class ToolSerializer(serializers.Serializer):
    label = serializers.CharField()
    description = serializers.CharField(allow_null=True, required=False)


class MCPServerStatusSerializer(serializers.Serializer):
    server_name = serializers.CharField()
    label = serializers.CharField()
    type = serializers.CharField()
    status = serializers.CharField()
    tool_names = serializers.ListField(child=serializers.CharField())


class ToolsResponseSerializer(serializers.Serializer):
    tools = ToolSerializer(many=True)
    mcp = MCPServerStatusSerializer(many=True, default=[])


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
    assistant_id = serializers.CharField(required=False, default="")


class ThreadListAPIView(APIView):
    def get(self, request: Request) -> Response:
        threads = list_threads(user=request.user)
        items = [
            {
                "id": t.id,
                "title": t.title,
                "assistant_id": t.assistant_id,
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
            messages = [Message(**m) for m in serializer.validated_data["messages"]]  # type: ignore[index, optional-subscript]
            assistant_id = request.data.get("assistant_id", "")  # type: ignore[union-attr]
            thread_id = create_thread(
                assistant_id=assistant_id,
                messages=messages,
                user=request.user,
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
        assistant_id = request.data.get("assistant_id")  # type: ignore[union-attr]
        if not assistant_id:
            return Response({"message": "assistant_id required"}, status=400)
        try:
            AssistantService.from_registry(assistant_id)
            thread = get_thread(thread_id, user=request.user)
            if thread is None:
                return Response({"message": "Thread not found"}, status=404)
            if thread.assistant_id:
                unlink_memories(thread.assistant_id, thread_id, user=request.user)
            update_thread(thread_id, metadata={"assistant_id": assistant_id}, user=request.user)
            link_memories(assistant_id, thread_id, user=request.user)
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


class ListAssistantsAPIView(APIView):
    def get(self, request: Request) -> Response:
        try:
            items = list_assistants(user=request.user)
            return Response(ListAssistantsSerializer({"assistants": items}).data)
        except PermissionDenied as e:
            return Response({"message": str(e)}, status=403)


class AssistantInfoAPIView(APIView):
    def get(self, request: Request, assistant_id: str) -> Response:
        try:
            assistant = AssistantService.from_registry(assistant_id)
            info = get_assistant_info(assistant_id, user=request.user)
            return Response(
                AssistantInfoSerializer(
                    {
                        "id": info.id,
                        "name": info.name,
                        "model": info.model,
                        "class_name": info.class_name,
                        "description": info.description,
                        "instructions": assistant.get_system_prompt(),
                        "file_upload": info.file_upload,
                    }
                ).data
            )
        except PermissionDenied as e:
            return Response({"message": str(e)}, status=403)
        except ValueError as e:
            return Response({"message": str(e)}, status=404)


class AssistantToolsAPIView(APIView):
    async def get(self, request: Request, assistant_id: str) -> Response:
        try:
            assistant = await AssistantService.get(assistant_id)
        except ValueError as e:
            return Response({"message": str(e)}, status=404)

        tools_data = []
        try:
            tool_objs = await assistant.get_tools()
            tools_data = [
                {
                    "label": getattr(t, "label", None) or t.name.replace("_", " ").title(),
                    "description": t.description or "",
                }
                for t in tool_objs
            ]
        except Exception:
            logger.exception("Failed to build tools for assistant %s", assistant_id)

        mcp_data = []
        try:
            mcp_status = await AssistantService.get_mcp_server_status(assistant, user=request.user)
            mcp_data = [
                {
                    "server_name": s.server_name,
                    "label": s.label,
                    "type": s.type,
                    "status": s.status,
                    "tool_names": s.tool_names,
                }
                for s in mcp_status
            ]
        except Exception:
            logger.exception("Failed to load MCP status for assistant %s", assistant_id)

        return Response(ToolsResponseSerializer({"tools": tools_data, "mcp": mcp_data}).data)


class ReindexAssistantAPIView(APIView):
    def post(self, request: Request, assistant_id: str) -> Response:
        memory_id = request.data.get("memory_id")  # type: ignore[union-attr]
        force_rebuild = request.data.get("force_rebuild", False)  # type: ignore[union-attr]
        try:
            assistant = AssistantService.from_registry(assistant_id)
            result = Assistant.reindex(assistant, memory_id, force_rebuild)  # type: ignore[arg-type]

            if not result:
                return Response(
                    ReindexResponseSerializer(
                        {
                            "success": False,
                            "message": "No RAG provider configured for this assistant",
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


class AssistantStatelessRunAPIView(APIView):
    """Stateless run"""

    async def post(self, request: Request, assistant_id: str) -> Response:
        try:
            serializer = ChatRequestSerializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            messages = [Message(**m) for m in serializer.validated_data["messages"]]  # type: ignore[index, optional-subscript]
            assistant = await AssistantService.get(assistant_id)
            chat_messages = assistant.protocol_handler.to_chat_messages(messages)
            result = await assistant.run(chat_messages, user=request.user)
            return Response({"result": result, "thread_id": None})
        except PermissionDenied as e:
            return Response({"message": str(e)}, status=403)
        except ValidationError as e:
            return Response({"message": str(e)}, status=400)
        except ValueError as e:
            return Response({"message": str(e)}, status=404)
        except NotImplementedError as e:
            return Response({"message": str(e)}, status=501)


class AssistantRunAPIView(APIView):
    """Synchronous JSON endpoint wrapping Assistant.run()"""

    async def post(self, request: Request, thread_id: str) -> Response:
        try:
            serializer = ChatRequestSerializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            messages = [Message(**m) for m in serializer.validated_data["messages"]]  # type: ignore[index, optional-subscript]
            assistant = await AssistantService.get_assistant(thread_id, user=request.user)
            chat_messages = assistant.protocol_handler.to_chat_messages(messages)
            result = await assistant.run(chat_messages, thread_id=thread_id, user=request.user)
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


class AssistantAPIView(View):
    async def post(self, request: HttpRequest, thread_id: str) -> StreamingHttpResponse:
        try:
            serializer = ChatRequestSerializer(data=json.loads(request.body))
            serializer.is_valid(raise_exception=True)
            messages = [Message(**m) for m in serializer.validated_data["messages"]]  # type: ignore[index, optional-subscript]
            assistant = await AssistantService.get_assistant(thread_id, user=request.user)
            return await assistant.as_view(messages, thread_id=thread_id, user=request.user)
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
# Runtime Assistants (DB-configured)
# ============================================================================


class AssistantUserInSerializer(serializers.Serializer):
    user_id = serializers.CharField()
    can_manage = serializers.BooleanField(default=False)


class AssistantGroupInSerializer(serializers.Serializer):
    group_id = serializers.IntegerField()
    can_manage = serializers.BooleanField(default=False)


class AssistantSettingsSerializer(serializers.Serializer):
    id = serializers.UUIDField(read_only=True)
    name = serializers.CharField()
    slug = serializers.SlugField()
    assistant = serializers.CharField(allow_blank=True, default="")
    model = serializers.CharField()
    system_prompt = serializers.CharField(allow_blank=True)
    tools = serializers.ListField(child=serializers.CharField(), default=list)
    mcp_servers = serializers.ListField(child=serializers.CharField(), default=list)
    suggestion_enabled = serializers.BooleanField(default=False)
    title_generation = serializers.BooleanField(default=True)
    max_history = serializers.IntegerField(allow_null=True, required=False)
    file_upload = serializers.BooleanField(default=False)
    active = serializers.BooleanField(default=True)
    created_at = serializers.DateTimeField(read_only=True)
    updated_at = serializers.DateTimeField(read_only=True)


class AssistantSettingsCreateSerializer(serializers.Serializer):
    name = serializers.CharField()
    slug = serializers.SlugField(default="")
    assistant = serializers.CharField(allow_blank=True, default="")
    model = serializers.CharField(default="gpt-4o")
    system_prompt = serializers.CharField(allow_blank=True, default="")
    tools = serializers.ListField(child=serializers.CharField(), default=list)
    mcp_servers = serializers.ListField(child=serializers.CharField(), default=list)
    users = AssistantUserInSerializer(many=True, required=False)
    groups = AssistantGroupInSerializer(many=True, required=False)
    suggestion_enabled = serializers.BooleanField(default=False)
    title_generation = serializers.BooleanField(default=True)
    max_history = serializers.IntegerField(allow_null=True, required=False)
    file_upload = serializers.BooleanField(default=False)


class AssistantSettingsUpdateSerializer(serializers.Serializer):
    name = serializers.CharField(required=False)
    assistant = serializers.CharField(allow_blank=True, required=False)
    model = serializers.CharField(required=False)
    system_prompt = serializers.CharField(allow_blank=True, required=False)
    tools = serializers.ListField(child=serializers.CharField(), required=False)
    mcp_servers = serializers.ListField(child=serializers.CharField(), required=False)
    users = AssistantUserInSerializer(many=True, required=False)
    groups = AssistantGroupInSerializer(many=True, required=False)
    suggestion_enabled = serializers.BooleanField(required=False)
    title_generation = serializers.BooleanField(required=False)
    max_history = serializers.IntegerField(allow_null=True, required=False)
    file_upload = serializers.BooleanField(required=False)
    active = serializers.BooleanField(required=False)


class RuntimeAssistantBasesAPIView(APIView):
    def get(self, request: Request) -> Response:
        from django_ai_sdk.assistants.config import get_runtime_assistant_bases

        data = [
            {"path": f"{cls.__module__}.{cls.__qualname__}", "name": cls.__name__}
            for cls in get_runtime_assistant_bases()
        ]
        return Response(data)


class RuntimeAssistantToolsAPIView(APIView):
    def get(self, request: Request) -> Response:
        from django_ai_sdk.assistants.config import get_tool_registry

        data = [{"key": k, "path": v} for k, v in get_tool_registry().items()]
        return Response(data)


class RuntimeAssistantListCreateAPIView(APIView):
    async def get(self, request: Request) -> Response:
        configs = await AssistantSettingsService.all()
        return Response(AssistantSettingsSerializer(configs, many=True).data)

    async def post(self, request: Request) -> Response:
        serializer = AssistantSettingsCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        skip_keys = {"users", "groups"}
        data = {k: v for k, v in serializer.validated_data.items() if k not in skip_keys}  # type: ignore[union-attr]
        try:
            config = await AssistantSettingsService.create(
                data,  # type: ignore[arg-type]
                user=request.user,
            )
        except Exception as e:
            return Response({"message": str(e)}, status=400)
        for entry in serializer.validated_data.get("users") or []:  # type: ignore[union-attr]
            try:
                await AssistantService.add_assistant_user(
                    str(config.id), entry["user_id"], entry.get("can_manage", False)
                )
            except Exception:
                pass
        for entry in serializer.validated_data.get("groups") or []:  # type: ignore[union-attr]
            try:
                await AssistantService.add_assistant_group(
                    str(config.id), entry["group_id"], entry.get("can_manage", False)
                )
            except Exception:
                pass
        return Response(AssistantSettingsSerializer(config).data)


class RuntimeAssistantDetailAPIView(APIView):
    async def get(self, request: Request, runtime_id: str) -> Response:
        try:
            config = await AssistantSettingsService.get(runtime_id)
            return Response(AssistantSettingsSerializer(config).data)
        except ValueError as e:
            return Response({"message": str(e)}, status=404)

    async def patch(self, request: Request, runtime_id: str) -> Response:
        serializer = AssistantSettingsUpdateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        skip_keys = {"users", "groups"}
        data = {k: v for k, v in serializer.validated_data.items() if k not in skip_keys}  # type: ignore[union-attr]
        try:
            config = await AssistantSettingsService.update(
                runtime_id,
                data,  # type: ignore[arg-type]
            )
        except ValueError as e:
            return Response({"message": str(e)}, status=404)
        except Exception as e:
            return Response({"message": str(e)}, status=400)
        for entry in serializer.validated_data.get("users") or []:  # type: ignore[union-attr]
            try:
                await AssistantService.add_assistant_user(
                    runtime_id, entry["user_id"], entry.get("can_manage", False)
                )
            except Exception:
                pass
        for entry in serializer.validated_data.get("groups") or []:  # type: ignore[union-attr]
            try:
                await AssistantService.add_assistant_group(runtime_id, entry["group_id"])
            except Exception:
                pass
        return Response(AssistantSettingsSerializer(config).data)

    async def delete(self, request: Request, runtime_id: str) -> Response:
        try:
            config = await AssistantSettingsService.delete(runtime_id)
            return Response(AssistantSettingsSerializer(config).data)
        except ValueError as e:
            return Response({"message": str(e)}, status=404)


# ── Assistant Users ───────────────────────────────────────────────────────────


class AssistantUserOutSerializer(serializers.Serializer):
    user_id = serializers.CharField()
    can_manage = serializers.BooleanField()
    created_at = serializers.CharField()


class AddAssistantUserInSerializer(serializers.Serializer):
    user_id = serializers.CharField()
    can_manage = serializers.BooleanField(default=False)


class UpdateAssistantUserInSerializer(serializers.Serializer):
    can_manage = serializers.BooleanField()


class AssistantUserListCreateAPIView(APIView):
    def get(self, request: Request, runtime_id: str) -> Response:
        try:
            users = list_assistant_users(runtime_id, user=request.user)
        except PermissionDenied as e:
            return Response({"detail": str(e)}, status=403)
        except ValueError as e:
            return Response({"detail": str(e)}, status=404)
        return Response(AssistantUserOutSerializer(users, many=True).data)

    def post(self, request: Request, runtime_id: str) -> Response:
        serializer = AddAssistantUserInSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            entry = add_assistant_user(
                runtime_id,
                serializer.validated_data["user_id"],  # type: ignore[index]
                serializer.validated_data.get("can_manage", False),  # type: ignore[union-attr]
                user=request.user,
            )
        except PermissionDenied as e:
            return Response({"detail": str(e)}, status=403)
        except ValueError as e:
            return Response({"detail": str(e)}, status=404)
        return Response(AssistantUserOutSerializer(entry).data)


class AssistantUserDetailAPIView(APIView):
    def patch(self, request: Request, runtime_id: str, user_id: str) -> Response:
        serializer = UpdateAssistantUserInSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            entry = update_assistant_user(
                runtime_id,
                user_id,
                serializer.validated_data["can_manage"],  # type: ignore[index]
                user=request.user,
            )
        except PermissionDenied as e:
            return Response({"detail": str(e)}, status=403)
        except ValueError as e:
            return Response({"detail": str(e)}, status=404)
        return Response(AssistantUserOutSerializer(entry).data)

    def delete(self, request: Request, runtime_id: str, user_id: str) -> Response:
        try:
            remove_assistant_user(runtime_id, user_id, user=request.user)
        except PermissionDenied as e:
            return Response({"detail": str(e)}, status=403)
        except ValueError as e:
            return Response({"detail": str(e)}, status=404)
        return Response(status=204)


# ── Assistant Groups ──────────────────────────────────────────────────────────


class AssistantGroupOutSerializer(serializers.Serializer):
    group_id = serializers.IntegerField()
    group_name = serializers.CharField(source="group.name")
    created_at = serializers.CharField()


class AddAssistantGroupInSerializer(serializers.Serializer):
    group_id = serializers.IntegerField()


class AssistantGroupListCreateAPIView(APIView):
    def get(self, request: Request, runtime_id: str) -> Response:
        try:
            groups = list_assistant_groups(runtime_id, user=request.user)
        except PermissionDenied as e:
            return Response({"detail": str(e)}, status=403)
        except ValueError as e:
            return Response({"detail": str(e)}, status=404)
        return Response(AssistantGroupOutSerializer(groups, many=True).data)

    def post(self, request: Request, runtime_id: str) -> Response:
        serializer = AddAssistantGroupInSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            entry = add_assistant_group(
                runtime_id,
                serializer.validated_data["group_id"],  # type: ignore[index]
                user=request.user,
            )
        except PermissionDenied as e:
            return Response({"detail": str(e)}, status=403)
        except ValueError as e:
            return Response({"detail": str(e)}, status=404)
        return Response(AssistantGroupOutSerializer(entry).data)


class AssistantGroupDetailAPIView(APIView):
    def delete(self, request: Request, runtime_id: str, group_id: int) -> Response:
        try:
            remove_assistant_group(runtime_id, group_id, user=request.user)
        except PermissionDenied as e:
            return Response({"detail": str(e)}, status=403)
        except ValueError as e:
            return Response({"detail": str(e)}, status=404)
        return Response(status=204)


urlpatterns = [
    path("assistants/", ListAssistantsAPIView.as_view(), name="assistant-list"),
    path(
        "assistants/<str:assistant_id>/",
        AssistantInfoAPIView.as_view(),
        name="assistant-info",
    ),
    path(
        "assistants/<str:assistant_id>/tools/",
        AssistantToolsAPIView.as_view(),
        name="assistant-tools",
    ),
    path(
        "assistants/<str:assistant_id>/reindex/",
        ReindexAssistantAPIView.as_view(),
        name="assistant-reindex",
    ),
    path(
        "assistants/<str:assistant_id>/run/",
        AssistantStatelessRunAPIView.as_view(),
        name="assistant-run",
    ),
    path("threads/", ThreadListAPIView.as_view(), name="thread-list"),
    path("threads/<str:thread_id>/", ThreadDetailAPIView.as_view(), name="thread-detail"),
    path(
        "threads/<str:thread_id>/file-meta/",
        ThreadFileMetaAPIView.as_view(),
        name="thread-file-meta",
    ),
    path("threads/", ThreadCreateAPIView.as_view(), name="thread-create"),
    path("threads/<str:thread_id>/delete/", ThreadDeleteAPIView.as_view(), name="thread-delete"),
    path(
        "threads/<str:thread_id>/message/",
        AssistantAPIView.as_view(),
        name="thread-message",
    ),
    path(
        "threads/<str:thread_id>/run/",
        AssistantRunAPIView.as_view(),
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
        "assistants/runtimes/bases/",
        RuntimeAssistantBasesAPIView.as_view(),
        name="runtime-assistant-bases",
    ),
    path(
        "assistants/runtimes/tools/",
        RuntimeAssistantToolsAPIView.as_view(),
        name="runtime-assistant-tools",
    ),
    path(
        "assistants/runtimes/",
        RuntimeAssistantListCreateAPIView.as_view(),
        name="runtime-assistant-list",
    ),
    path(
        "assistants/runtimes/<str:runtime_id>/",
        RuntimeAssistantDetailAPIView.as_view(),
        name="runtime-assistant-detail",
    ),
    path(
        "assistants/runtimes/<str:runtime_id>/users/",
        AssistantUserListCreateAPIView.as_view(),
        name="runtime-assistant-user-list",
    ),
    path(
        "assistants/runtimes/<str:runtime_id>/users/<str:user_id>/",
        AssistantUserDetailAPIView.as_view(),
        name="runtime-assistant-user-detail",
    ),
    path(
        "assistants/runtimes/<str:runtime_id>/groups/",
        AssistantGroupListCreateAPIView.as_view(),
        name="runtime-assistant-group-list",
    ),
    path(
        "assistants/runtimes/<str:runtime_id>/groups/<int:group_id>/",
        AssistantGroupDetailAPIView.as_view(),
        name="runtime-assistant-group-detail",
    ),
]
