import json
from typing import Any

from django.http import HttpRequest, StreamingHttpResponse
from django.urls import path
from django.views import View
from django_ai_sdk import Assistant
from django_ai_sdk.assistants.services import (
    AssistantService,
    get_assistant_info,
    list_assistants,
)
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


class ThreadListItemSerializer(serializers.Serializer):
    id = serializers.CharField()
    title = serializers.CharField()
    assistant_id = serializers.CharField()
    created_at = serializers.CharField()
    updated_at = serializers.CharField()
    message_count = serializers.IntegerField()


class ThreadListResponseSerializer(serializers.Serializer):
    threads = ThreadListItemSerializer(many=True)


class ThreadDetailSerializer(serializers.Serializer):
    thread = serializers.DictField()
    messages = serializers.ListField()


class ThreadFileMetaSerializer(serializers.Serializer):
    file_count = serializers.IntegerField()
    file_memory_id = serializers.CharField(allow_null=True)


class CreateThreadResponseSerializer(serializers.Serializer):
    thread_id = serializers.CharField(allow_null=True)


class FeedbackResponseSerializer(serializers.Serializer):
    id = serializers.CharField()
    user_id = serializers.CharField(allow_null=True)
    rating = serializers.IntegerField()
    feedback = serializers.CharField()
    created_at = serializers.CharField(allow_null=True)


class MessageResponseSerializer(serializers.Serializer):
    id = serializers.CharField()
    is_deleted = serializers.BooleanField(allow_null=True)
    feedbacks = FeedbackResponseSerializer(many=True, default=[])


class DeleteAllThreadsResponseSerializer(serializers.Serializer):
    success = serializers.BooleanField()
    deleted_count = serializers.IntegerField()


class AssistantInfoSerializer(serializers.Serializer):
    id = serializers.CharField()
    name = serializers.CharField()
    model = serializers.CharField()


class ListAssistantsSerializer(serializers.Serializer):
    assistants = AssistantInfoSerializer(many=True)


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
            messages = [Message(**m) for m in serializer.validated_data["messages"]]
            assistant_id = request.data.get("assistant_id", "")
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
            return Response(ThreadDetailSerializer(data).data)
        except PermissionDenied as e:
            return Response({"message": str(e)}, status=403)
        except ValueError as e:
            return Response({"message": str(e)}, status=404)

    def patch(self, request: Request, thread_id: str) -> Response:
        assistant_id = request.data.get("assistant_id")
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

        if "rating" not in request.data:
            return Response({"message": "rating is required"}, status=400)
        rating = request.data.get("rating")
        feedback_text = request.data.get("feedback", "")
        try:
            rate_message(thread_id, message_id, rating, feedback=feedback_text, user=request.user)

            return Response(
                MessageResponseSerializer(
                    {
                        "id": message_id,
                        "is_deleted": False,
                    }
                ).data
            )
        except PermissionDenied as e:
            return Response({"message": str(e)}, status=403)
        except ValueError as e:
            return Response({"message": str(e)}, status=404)


class DeleteMessageAPIView(APIView):
    def post(self, request: Request, thread_id: str, message_id: str) -> Response:
        try:
            delete_message(thread_id, message_id, user=request.user)
            return Response(
                MessageResponseSerializer(
                    {
                        "id": message_id,
                        "is_deleted": True,
                    }
                ).data
            )
        except PermissionDenied as e:
            return Response({"message": str(e)}, status=403)
        except ValueError as e:
            return Response({"message": str(e)}, status=404)


class RestoreMessageAPIView(APIView):
    def post(self, request: Request, thread_id: str, message_id: str) -> Response:
        try:
            restore_message(thread_id, message_id, user=request.user)
            return Response(
                MessageResponseSerializer(
                    {
                        "id": message_id,
                        "is_deleted": False,
                    }
                ).data
            )
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
            info = get_assistant_info(assistant_id, user=request.user)
            return Response(AssistantInfoSerializer(info).data)
        except PermissionDenied as e:
            return Response({"message": str(e)}, status=403)
        except ValueError as e:
            return Response({"message": str(e)}, status=404)


class ReindexAssistantAPIView(APIView):
    def post(self, request: Request, assistant_id: str) -> Response:
        memory_id = request.data.get("memory_id")
        force_rebuild = request.data.get("force_rebuild", False)
        try:
            assistant = AssistantService.from_registry(assistant_id)
            result = Assistant.reindex(assistant, memory_id, force_rebuild)

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

            return Response(
                ReindexResponseSerializer(
                    {
                        "success": True,
                        "message": message,
                    }
                ).data
            )
        except ValueError as e:
            return Response({"message": str(e)}, status=404)


class AssistantAPIView(View):
    async def post(self, request: HttpRequest, thread_id: str) -> StreamingHttpResponse:
        try:
            serializer = ChatRequestSerializer(data=json.loads(request.body))
            serializer.is_valid(raise_exception=True)
            messages = [Message(**m) for m in serializer.validated_data["messages"]]
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


urlpatterns = [
    path("assistants/", ListAssistantsAPIView.as_view(), name="assistant-list"),
    path(
        "assistants/<str:assistant_id>/",
        AssistantInfoAPIView.as_view(),
        name="assistant-info",
    ),
    path(
        "assistants/<str:assistant_id>/reindex/",
        ReindexAssistantAPIView.as_view(),
        name="assistant-reindex",
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
]
