from django.urls import path
from django_ai_sdk.assistants.services import WebAssistantService
from rest_framework import serializers
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView


class WebAssistantSerializer(serializers.Serializer):
    id = serializers.UUIDField(read_only=True)
    name = serializers.CharField()
    slug = serializers.SlugField()
    base_class = serializers.CharField(allow_blank=True, default="")
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


class WebAssistantCreateSerializer(serializers.Serializer):
    name = serializers.CharField()
    slug = serializers.SlugField(default="")
    base_class = serializers.CharField(allow_blank=True, default="")
    model = serializers.CharField(default="gpt-4o")
    system_prompt = serializers.CharField(allow_blank=True, default="")
    tools = serializers.ListField(child=serializers.CharField(), default=list)
    mcp_servers = serializers.ListField(child=serializers.CharField(), default=list)
    suggestion_enabled = serializers.BooleanField(default=False)
    title_generation = serializers.BooleanField(default=True)
    max_history = serializers.IntegerField(allow_null=True, required=False)
    file_upload = serializers.BooleanField(default=False)


class WebAssistantUpdateSerializer(serializers.Serializer):
    name = serializers.CharField(required=False)
    base_class = serializers.CharField(allow_blank=True, required=False)
    model = serializers.CharField(required=False)
    system_prompt = serializers.CharField(allow_blank=True, required=False)
    tools = serializers.ListField(child=serializers.CharField(), required=False)
    mcp_servers = serializers.ListField(child=serializers.CharField(), required=False)
    suggestion_enabled = serializers.BooleanField(required=False)
    title_generation = serializers.BooleanField(required=False)
    max_history = serializers.IntegerField(allow_null=True, required=False)
    file_upload = serializers.BooleanField(required=False)
    active = serializers.BooleanField(required=False)


class WebAssistantBasesAPIView(APIView):
    def get(self, request: Request) -> Response:
        from django_ai_sdk.web_assistant.config import get_web_assistant_bases

        data = [
            {"path": f"{cls.__module__}.{cls.__qualname__}", "name": cls.__name__}
            for cls in get_web_assistant_bases()
        ]
        return Response(data)


class WebAssistantToolsAPIView(APIView):
    def get(self, request: Request) -> Response:
        from django_ai_sdk.web_assistant.config import get_tool_registry

        data = [{"key": k, "path": v} for k, v in get_tool_registry().items()]
        return Response(data)


class WebAssistantListCreateAPIView(APIView):
    async def get(self, request: Request) -> Response:
        configs = await WebAssistantService.all()
        return Response(WebAssistantSerializer(configs, many=True).data)

    async def post(self, request: Request) -> Response:
        serializer = WebAssistantCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            config = await WebAssistantService.create(
                serializer.validated_data,  # type: ignore[arg-type]
                user=request.user,
            )
            return Response(WebAssistantSerializer(config).data, status=201)
        except Exception as e:
            return Response({"message": str(e)}, status=400)


class WebAssistantDetailAPIView(APIView):
    async def get(self, request: Request, assistant_id: str) -> Response:
        try:
            config = await WebAssistantService.get(assistant_id)
            return Response(WebAssistantSerializer(config).data)
        except ValueError as e:
            return Response({"message": str(e)}, status=404)

    async def patch(self, request: Request, assistant_id: str) -> Response:
        serializer = WebAssistantUpdateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            config = await WebAssistantService.update(
                assistant_id,
                serializer.validated_data,  # type: ignore[arg-type]
            )
            return Response(WebAssistantSerializer(config).data)
        except ValueError as e:
            return Response({"message": str(e)}, status=404)
        except Exception as e:
            return Response({"message": str(e)}, status=400)

    async def delete(self, request: Request, assistant_id: str) -> Response:
        try:
            config = await WebAssistantService.delete(assistant_id)
            return Response(WebAssistantSerializer(config).data)
        except ValueError as e:
            return Response({"message": str(e)}, status=404)


urlpatterns = [
    path("web-assistants/bases/", WebAssistantBasesAPIView.as_view(), name="web-assistant-bases"),
    path("web-assistants/tools/", WebAssistantToolsAPIView.as_view(), name="web-assistant-tools"),
    path("web-assistants/", WebAssistantListCreateAPIView.as_view(), name="web-assistant-list"),
    path(
        "web-assistants/<str:assistant_id>/",
        WebAssistantDetailAPIView.as_view(),
        name="web-assistant-detail",
    ),
]
