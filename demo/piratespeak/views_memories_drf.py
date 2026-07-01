from __future__ import annotations

from typing import TYPE_CHECKING

from django.urls import path
from django_ai_sdk.memories.services import (
    add_memory_user,
    bulk_connect_memories,
    create_memory,
    delete_document,
    delete_memory,
    delete_thread_file,
    disconnect_memory_from_thread,
    get_chunk_content,
    get_document,
    get_document_status,
    get_memory,
    link_memory_to_thread,
    list_documents,
    list_memories,
    list_memory_users,
    list_thread_files,
    list_thread_memories,
    remove_memory_user,
    toggle_memory_active,
    unlink_memory_from_thread,
    update_memory,
    update_memory_user,
    upload_document,
    upload_thread_file,
)
from django_ai_sdk.permissions import ConflictError, PermissionDenied
from rest_framework import serializers
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

if TYPE_CHECKING:
    from rest_framework.request import Request


class MemoryUserInSerializer(serializers.Serializer):
    user_id = serializers.CharField()
    can_manage = serializers.BooleanField(default=False)


class MemoryInSerializer(serializers.Serializer):
    name = serializers.CharField()
    slug = serializers.CharField(required=False, default="")
    description = serializers.CharField(required=False, default="")
    is_public = serializers.BooleanField(required=False, default=True)
    users = MemoryUserInSerializer(many=True, required=False, default=list)


class MemoryOutSerializer(serializers.Serializer):
    id = serializers.CharField()
    name = serializers.CharField()
    slug = serializers.CharField()
    description = serializers.CharField()
    is_public = serializers.BooleanField()
    document_count = serializers.IntegerField()
    created_at = serializers.CharField()
    updated_at = serializers.CharField()


class DocumentOutSerializer(serializers.Serializer):
    id = serializers.CharField()
    file = serializers.CharField()
    content = serializers.CharField()
    extraction = serializers.DictField(allow_null=True)
    file_name = serializers.CharField()
    file_size = serializers.IntegerField()
    content_type = serializers.CharField()
    file_extension = serializers.CharField()
    created_at = serializers.CharField()
    updated_at = serializers.CharField()


class ThreadMemoryOutSerializer(serializers.Serializer):
    id = serializers.CharField()
    name = serializers.CharField()
    description = serializers.CharField()
    document_count = serializers.IntegerField()
    active = serializers.BooleanField()
    created_at = serializers.CharField()


class BulkConnectMemoriesInSerializer(serializers.Serializer):
    memory_ids = serializers.ListField(child=serializers.CharField())


class ToggleMemoryActiveInSerializer(serializers.Serializer):
    active = serializers.BooleanField()


class MemoryUserOutSerializer(serializers.Serializer):
    user_id = serializers.CharField()
    can_manage = serializers.BooleanField()
    created_at = serializers.CharField()


class AddMemoryUserInSerializer(serializers.Serializer):
    user_id = serializers.CharField()
    can_manage = serializers.BooleanField(default=False)


class UpdateMemoryUserInSerializer(serializers.Serializer):
    can_manage = serializers.BooleanField()


class SourceContentSerializer(serializers.Serializer):
    content = serializers.CharField()


class DocumentUploadResponseSerializer(serializers.Serializer):
    id = serializers.CharField()
    status = serializers.CharField()
    processing_step = serializers.CharField(allow_null=True, required=False)
    task_id = serializers.CharField(allow_null=True, required=False)


class DocumentStatusOutSerializer(serializers.Serializer):
    id = serializers.CharField()
    status = serializers.CharField()
    error = serializers.CharField()
    processing_step = serializers.CharField(allow_null=True, required=False)
    task = serializers.DictField(allow_null=True, required=False)


class MemoryListCreateAPIView(APIView):
    def get(self, request: Request) -> Response:
        limit = int(request.query_params.get("limit", 100))
        offset = int(request.query_params.get("offset", 0))
        try:
            memories = list_memories(user=request.user, limit=limit, offset=offset)
        except PermissionDenied as e:
            return Response({"detail": str(e)}, status=403)
        return Response(MemoryOutSerializer(memories, many=True).data)

    def post(self, request: Request) -> Response:
        serializer = MemoryInSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            memory = create_memory(
                name=serializer.validated_data["name"],  # type: ignore[index, optional-subscript]
                slug=serializer.validated_data.get("slug", ""),  # type: ignore[union-attr]
                description=serializer.validated_data.get("description", ""),  # type: ignore[union-attr]
                is_public=serializer.validated_data.get("is_public", True),  # type: ignore[union-attr]
                user=request.user,
            )
        except PermissionDenied as e:
            return Response({"detail": str(e)}, status=403)
        for owner in serializer.validated_data.get("users") or []:  # type: ignore[union-attr]
            try:
                add_memory_user(
                    str(memory.id), owner["user_id"], owner["can_manage"], user=request.user
                )
            except Exception:
                pass
        return Response(MemoryOutSerializer(memory).data)


class MemoryDetailAPIView(APIView):
    def get(self, request: Request, memory_id: str) -> Response:
        try:
            memory = get_memory(memory_id, user=request.user)
        except PermissionDenied as e:
            return Response({"detail": str(e)}, status=403)
        return Response(MemoryOutSerializer(memory).data)

    def put(self, request: Request, memory_id: str) -> Response:
        serializer = MemoryInSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            memory = update_memory(
                memory_id=memory_id,
                name=serializer.validated_data["name"],  # type: ignore[index, optional-subscript]
                description=serializer.validated_data.get("description", ""),  # type: ignore[union-attr]
                is_public=serializer.validated_data.get("is_public", True),  # type: ignore[union-attr]
                user=request.user,
            )
        except PermissionDenied as e:
            return Response({"detail": str(e)}, status=403)
        for owner in serializer.validated_data.get("users") or []:  # type: ignore[union-attr]
            try:
                add_memory_user(memory_id, owner["user_id"], owner["can_manage"], user=request.user)
            except Exception:
                pass
        return Response(MemoryOutSerializer(memory).data)

    def delete(self, request: Request, memory_id: str) -> Response:
        try:
            delete_memory(memory_id, user=request.user)
        except PermissionDenied as e:
            return Response({"detail": str(e)}, status=403)
        return Response(status=204)


class DocumentListCreateAPIView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def get(self, request: Request, memory_id: str) -> Response:
        limit = int(request.query_params.get("limit", 100))
        offset = int(request.query_params.get("offset", 0))
        try:
            documents = list_documents(memory_id, user=request.user, limit=limit, offset=offset)
        except PermissionDenied as e:
            return Response({"detail": str(e)}, status=403)
        return Response(DocumentOutSerializer(documents, many=True).data)

    def post(self, request: Request, memory_id: str) -> Response:
        uploaded_file = request.FILES.get("file")  # type: ignore[union-attr]
        if not uploaded_file:
            return Response({"detail": "file is required"}, status=400)
        try:
            result = upload_document(memory_id, uploaded_file, user=request.user)
        except PermissionDenied as e:
            return Response({"detail": str(e)}, status=403)
        except ConflictError as e:
            return Response({"detail": str(e)}, status=409)
        return Response(DocumentUploadResponseSerializer(result).data, status=202)


class DocumentDetailAPIView(APIView):
    def get(self, request: Request, memory_id: str, doc_id: str) -> Response:
        try:
            document = get_document(memory_id, doc_id, user=request.user)
        except PermissionDenied as e:
            return Response({"detail": str(e)}, status=403)
        return Response(DocumentOutSerializer(document).data)

    def delete(self, request: Request, memory_id: str, doc_id: str) -> Response:
        try:
            delete_document(memory_id, doc_id, user=request.user)
        except PermissionDenied as e:
            return Response({"detail": str(e)}, status=403)
        return Response(status=204)


class LinkMemoryThreadAPIView(APIView):
    def post(self, request: Request, memory_id: str, thread_id: str) -> Response:
        try:
            link_memory_to_thread(memory_id, thread_id, user=request.user)
        except PermissionDenied as e:
            return Response({"detail": str(e)}, status=403)
        return Response(status=204)

    def delete(self, request: Request, memory_id: str, thread_id: str) -> Response:
        try:
            unlink_memory_from_thread(memory_id, thread_id, user=request.user)
        except PermissionDenied as e:
            return Response({"detail": str(e)}, status=403)
        return Response(status=204)


class ThreadMemoryListAPIView(APIView):
    def get(self, request: Request, thread_id: str) -> Response:
        limit = int(request.query_params.get("limit", 100))
        offset = int(request.query_params.get("offset", 0))
        try:
            memories = list_thread_memories(
                thread_id, user=request.user, limit=limit, offset=offset
            )
        except PermissionDenied as e:
            return Response({"detail": str(e)}, status=403)
        return Response(ThreadMemoryOutSerializer(memories, many=True).data)


class ThreadMemoryBulkConnectAPIView(APIView):
    def post(self, request: Request, thread_id: str) -> Response:
        serializer = BulkConnectMemoriesInSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            memories = bulk_connect_memories(
                thread_id,
                serializer.validated_data["memory_ids"],  # type: ignore[index, optional-subscript]
                user=request.user,
            )
        except PermissionDenied as e:
            return Response({"detail": str(e)}, status=403)
        return Response(ThreadMemoryOutSerializer(memories, many=True).data)


class ThreadMemoryToggleAPIView(APIView):
    def patch(self, request: Request, thread_id: str, memory_id: str) -> Response:
        serializer = ToggleMemoryActiveInSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            result = toggle_memory_active(
                thread_id,
                memory_id,
                serializer.validated_data["active"],  # type: ignore[index, optional-subscript]
                user=request.user,
            )
        except PermissionDenied as e:
            return Response({"detail": str(e)}, status=403)
        return Response(ThreadMemoryOutSerializer(result).data)

    def delete(self, request: Request, thread_id: str, memory_id: str) -> Response:
        try:
            disconnect_memory_from_thread(thread_id, memory_id, user=request.user)
        except PermissionDenied as e:
            return Response({"detail": str(e)}, status=403)
        return Response(status=204)


class ThreadFileListCreateAPIView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def get(self, request: Request, thread_id: str) -> Response:
        limit = int(request.query_params.get("limit", 100))
        offset = int(request.query_params.get("offset", 0))
        files = list_thread_files(thread_id, user=request.user, limit=limit, offset=offset)
        return Response(DocumentOutSerializer(files, many=True).data)

    def post(self, request: Request, thread_id: str) -> Response:
        uploaded_file = request.FILES.get("file")  # type: ignore[union-attr]
        if not uploaded_file:
            return Response({"detail": "file is required"}, status=400)
        try:
            result = upload_thread_file(thread_id, uploaded_file, user=request.user)
        except ConflictError as e:
            return Response({"detail": str(e)}, status=409)
        return Response(DocumentUploadResponseSerializer(result).data, status=202)


class ThreadFileDetailAPIView(APIView):
    def delete(self, request: Request, thread_id: str, doc_id: str) -> Response:
        delete_thread_file(thread_id, doc_id, user=request.user)
        return Response(status=204)


class DocumentStatusAPIView(APIView):
    def get(self, request: Request, doc_id: str) -> Response:
        status = get_document_status(doc_id, user=request.user)
        return Response(DocumentStatusOutSerializer(status).data)


class MemoryUserListCreateAPIView(APIView):
    def get(self, request: Request, memory_id: str) -> Response:
        limit = int(request.query_params.get("limit", 100))
        offset = int(request.query_params.get("offset", 0))
        try:
            users = list_memory_users(memory_id, user=request.user, limit=limit, offset=offset)
        except PermissionDenied as e:
            return Response({"detail": str(e)}, status=403)
        except ValueError as e:
            return Response({"detail": str(e)}, status=404)
        return Response(MemoryUserOutSerializer(users, many=True).data)

    def post(self, request: Request, memory_id: str) -> Response:
        serializer = AddMemoryUserInSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            user = add_memory_user(
                memory_id,
                serializer.validated_data["user_id"],  # type: ignore[index, optional-subscript]
                serializer.validated_data.get("can_manage", False),  # type: ignore[union-attr]
                user=request.user,
            )
        except PermissionDenied as e:
            return Response({"detail": str(e)}, status=403)
        except ValueError as e:
            return Response({"detail": str(e)}, status=404)
        return Response(MemoryUserOutSerializer(user).data)


class SourceContentAPIView(APIView):
    def get(self, request: Request, entry_id: str, chunk_id: str) -> Response:
        content = get_chunk_content(entry_id, chunk_id or None, user=request.user)
        if content is None:
            return Response({"detail": f"Entry not found: {entry_id}"}, status=404)
        return Response(SourceContentSerializer({"content": content}).data)


class MemoryUserDetailAPIView(APIView):
    def patch(self, request: Request, memory_id: str, user_id: str) -> Response:
        serializer = UpdateMemoryUserInSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            user = update_memory_user(
                memory_id,
                user_id,
                serializer.validated_data["can_manage"],  # type: ignore[index, optional-subscript]
                user=request.user,
            )
        except PermissionDenied as e:
            return Response({"detail": str(e)}, status=403)
        except ValueError as e:
            return Response({"detail": str(e)}, status=404)
        return Response(MemoryUserOutSerializer(user).data)

    def delete(self, request: Request, memory_id: str, user_id: str) -> Response:
        try:
            remove_memory_user(memory_id, user_id, user=request.user)
        except PermissionDenied as e:
            return Response({"detail": str(e)}, status=403)
        except ValueError as e:
            return Response({"detail": str(e)}, status=404)
        return Response(status=204)


urlpatterns = [
    path("memories/", MemoryListCreateAPIView.as_view(), name="memory-list-create"),
    path("memories/<str:memory_id>/", MemoryDetailAPIView.as_view(), name="memory-detail"),
    path(
        "memories/<str:memory_id>/documents/",
        DocumentListCreateAPIView.as_view(),
        name="document-list-create",
    ),
    path(
        "memories/<str:memory_id>/documents/<str:doc_id>/",
        DocumentDetailAPIView.as_view(),
        name="document-detail",
    ),
    path(
        "memories/<str:memory_id>/link/<str:thread_id>/",
        LinkMemoryThreadAPIView.as_view(),
        name="memory-link-thread",
    ),
    path(
        "memories/thread/<str:thread_id>/",
        ThreadMemoryListAPIView.as_view(),
        name="thread-memory-list",
    ),
    path(
        "memories/thread/<str:thread_id>/bulk/",
        ThreadMemoryBulkConnectAPIView.as_view(),
        name="thread-memory-bulk",
    ),
    path(
        "memories/thread/<str:thread_id>/files/",
        ThreadFileListCreateAPIView.as_view(),
        name="thread-file-list-create",
    ),
    path(
        "memories/thread/<str:thread_id>/files/<str:doc_id>/",
        ThreadFileDetailAPIView.as_view(),
        name="thread-file-detail",
    ),
    path(
        "memories/thread/<str:thread_id>/<str:memory_id>/",
        ThreadMemoryToggleAPIView.as_view(),
        name="thread-memory-toggle",
    ),
    path(
        "memories/<str:memory_id>/users/",
        MemoryUserListCreateAPIView.as_view(),
        name="memory-user-list-create",
    ),
    path(
        "memories/<str:memory_id>/users/<str:user_id>/",
        MemoryUserDetailAPIView.as_view(),
        name="memory-user-detail",
    ),
    path(
        "memories/source/<str:entry_id>/<str:chunk_id>/",
        SourceContentAPIView.as_view(),
        name="memory-source-content",
    ),
    path(
        "memories/documents/<str:doc_id>/status/",
        DocumentStatusAPIView.as_view(),
        name="document-status",
    ),
]
