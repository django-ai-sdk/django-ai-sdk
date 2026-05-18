from django.urls import path
from django_ai_sdk.memories.services import (
    bulk_connect_memories,
    create_memory,
    delete_document,
    delete_memory,
    delete_thread_file,
    disconnect_memory_from_thread,
    get_document,
    get_memory,
    link_memory_to_thread,
    list_documents,
    list_memories,
    list_thread_files,
    list_thread_memories,
    toggle_memory_active,
    unlink_memory_from_thread,
    update_memory,
    upload_document,
    upload_thread_file,
)
from rest_framework import serializers
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView


class MemoryInSerializer(serializers.Serializer):
    name = serializers.CharField()
    slug = serializers.CharField(required=False, default="")
    description = serializers.CharField(required=False, default="")


class MemoryOutSerializer(serializers.Serializer):
    id = serializers.CharField()
    name = serializers.CharField()
    slug = serializers.CharField()
    description = serializers.CharField()
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


class MemoryListCreateAPIView(APIView):
    def get(self, request: Request) -> Response:
        memories = list_memories()
        return Response(MemoryOutSerializer(memories, many=True).data)

    def post(self, request: Request) -> Response:
        serializer = MemoryInSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        memory = create_memory(
            name=serializer.validated_data["name"],
            slug=serializer.validated_data.get("slug", ""),
            description=serializer.validated_data.get("description", ""),
        )
        return Response(MemoryOutSerializer(memory).data)


class MemoryDetailAPIView(APIView):
    def get(self, request: Request, memory_id: str) -> Response:
        memory = get_memory(memory_id)
        return Response(MemoryOutSerializer(memory).data)

    def put(self, request: Request, memory_id: str) -> Response:
        serializer = MemoryInSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        memory = update_memory(
            memory_id=memory_id,
            name=serializer.validated_data["name"],
            description=serializer.validated_data.get("description", ""),
        )
        return Response(MemoryOutSerializer(memory).data)

    def delete(self, request: Request, memory_id: str) -> Response:
        delete_memory(memory_id)
        return Response(status=204)


class DocumentListCreateAPIView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def get(self, request: Request, memory_id: str) -> Response:
        documents = list_documents(memory_id)
        return Response(DocumentOutSerializer(documents, many=True).data)

    def post(self, request: Request, memory_id: str) -> Response:
        uploaded_file = request.FILES.get("file")
        if not uploaded_file:
            return Response({"detail": "file is required"}, status=400)
        result = upload_document(memory_id, uploaded_file)
        if isinstance(result, tuple):
            return Response(result[1], status=result[0])
        return Response(DocumentOutSerializer(result).data)


class DocumentDetailAPIView(APIView):
    def get(self, request: Request, memory_id: str, doc_id: str) -> Response:
        document = get_document(memory_id, doc_id)
        return Response(DocumentOutSerializer(document).data)

    def delete(self, request: Request, memory_id: str, doc_id: str) -> Response:
        delete_document(memory_id, doc_id)
        return Response(status=204)


class LinkMemoryThreadAPIView(APIView):
    def post(self, request: Request, memory_id: str, thread_id: str) -> Response:
        link_memory_to_thread(memory_id, thread_id)
        return Response(status=204)

    def delete(self, request: Request, memory_id: str, thread_id: str) -> Response:
        unlink_memory_from_thread(memory_id, thread_id)
        return Response(status=204)


class ThreadMemoryListAPIView(APIView):
    def get(self, request: Request, thread_id: str) -> Response:
        memories = list_thread_memories(thread_id)
        return Response(ThreadMemoryOutSerializer(memories, many=True).data)


class ThreadMemoryBulkConnectAPIView(APIView):
    def post(self, request: Request, thread_id: str) -> Response:
        serializer = BulkConnectMemoriesInSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        memories = bulk_connect_memories(thread_id, serializer.validated_data["memory_ids"])
        return Response(ThreadMemoryOutSerializer(memories, many=True).data)


class ThreadMemoryToggleAPIView(APIView):
    def patch(self, request: Request, thread_id: str, memory_id: str) -> Response:
        serializer = ToggleMemoryActiveInSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        result = toggle_memory_active(thread_id, memory_id, serializer.validated_data["active"])
        return Response(ThreadMemoryOutSerializer(result).data)

    def delete(self, request: Request, thread_id: str, memory_id: str) -> Response:
        disconnect_memory_from_thread(thread_id, memory_id)
        return Response(status=204)


class ThreadFileListCreateAPIView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def get(self, request: Request, thread_id: str) -> Response:
        files = list_thread_files(thread_id)
        return Response(DocumentOutSerializer(files, many=True).data)

    def post(self, request: Request, thread_id: str) -> Response:
        uploaded_file = request.FILES.get("file")
        if not uploaded_file:
            return Response({"detail": "file is required"}, status=400)
        result = upload_thread_file(thread_id, uploaded_file)
        if isinstance(result, tuple):
            return Response(result[1], status=result[0])
        return Response(DocumentOutSerializer(result).data)


class ThreadFileDetailAPIView(APIView):
    def delete(self, request: Request, thread_id: str, doc_id: str) -> Response:
        delete_thread_file(thread_id, doc_id)
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
]
