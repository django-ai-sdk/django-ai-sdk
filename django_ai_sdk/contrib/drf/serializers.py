"""DRF serializers for the SDK viewsets.

Request serializers validate input (DRF's 400 format, browsable-API forms, schema
generation). Response serializers render what the services return: pydantic models are
read by attribute like any object, model instances through ``ModelSerializer``.

Swap a request serializer by overriding ``serializer_classes`` on a viewset subclass.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError
from rest_framework import serializers

from django_ai_sdk.agents.models import AgentSettings
from django_ai_sdk.workflows.models import WorkflowRun, WorkflowRunStep, WorkflowSettings
from django_ai_sdk.workflows.schemas import WorkflowDefinition


class PydanticField(serializers.Field):
    """Read-only: renders a nested pydantic value (or dict of them) as JSON."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("read_only", True)
        super().__init__(**kwargs)

    def to_representation(self, value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, dict):
            return {k: self.to_representation(v) for k, v in value.items()}
        return value


# --- Requests ---------------------------------------------------------------


class MessagePartSerializer(serializers.Serializer):
    """A Vercel UI message part. Keys keep the client's camelCase spelling."""

    type = serializers.CharField()
    text = serializers.CharField(required=False, allow_null=True, allow_blank=True)
    # file parts
    url = serializers.CharField(required=False, allow_null=True)
    mediaType = serializers.CharField(required=False, allow_null=True)  # noqa: N815
    filename = serializers.CharField(required=False, allow_null=True)
    providerMetadata = serializers.DictField(required=False, allow_null=True)  # noqa: N815


class MessageSerializer(serializers.Serializer):
    role = serializers.CharField()
    parts = MessagePartSerializer(many=True)
    id = serializers.CharField(required=False, allow_null=True)


class ChatRequestSerializer(serializers.Serializer):
    """A Vercel UI chat request: the conversation so far and, optionally, the agent."""

    messages = MessageSerializer(many=True)
    agent_id = serializers.CharField(required=False, allow_null=True, allow_blank=True)
    id = serializers.CharField(required=False, allow_null=True)
    trigger = serializers.CharField(required=False, allow_null=True)


class RateMessageSerializer(serializers.Serializer):
    rating = serializers.IntegerField(
        required=False, allow_null=True, help_text="1 good, -1 bad, null to unrate"
    )
    feedback = serializers.CharField(required=False, allow_blank=True, default="")


class AgentSwitchSerializer(serializers.Serializer):
    agent_id = serializers.CharField()


class MemorySerializer(serializers.Serializer):
    name = serializers.CharField()
    slug = serializers.CharField(required=False, allow_blank=True, default="")
    description = serializers.CharField(required=False, allow_blank=True, default="")
    is_public = serializers.BooleanField(required=False, default=True)


class BulkConnectSerializer(serializers.Serializer):
    memory_ids = serializers.ListField(child=serializers.CharField())


class ToggleActiveSerializer(serializers.Serializer):
    active = serializers.BooleanField()


class AddUserSerializer(serializers.Serializer):
    user_id = serializers.CharField()
    can_manage = serializers.BooleanField(required=False, default=False)


class UpdateUserSerializer(serializers.Serializer):
    can_manage = serializers.BooleanField()


class AddGroupSerializer(serializers.Serializer):
    group_id = serializers.IntegerField()
    can_manage = serializers.BooleanField(required=False, default=False)


class RuntimeAgentCreateSerializer(serializers.Serializer):
    name = serializers.CharField()
    slug = serializers.CharField(required=False, allow_blank=True, default="")
    agent = serializers.CharField(required=False, allow_blank=True, default="")
    model = serializers.CharField(required=False, default="gpt-4o")
    system_prompt = serializers.CharField(required=False, allow_blank=True, default="")
    tools = serializers.ListField(child=serializers.CharField(), required=False, default=list)
    integrations = serializers.ListField(
        child=serializers.CharField(), required=False, default=list
    )
    memories = serializers.ListField(child=serializers.CharField(), required=False, default=list)
    users = AddUserSerializer(many=True, required=False, default=list)
    groups = AddGroupSerializer(many=True, required=False, default=list)
    suggestion_enabled = serializers.BooleanField(required=False, default=False)
    title_generation = serializers.BooleanField(required=False, default=True)
    max_history = serializers.IntegerField(required=False, allow_null=True, default=None)
    file_upload = serializers.BooleanField(required=False, default=False)
    vision = serializers.BooleanField(required=False, default=False)
    required_tools = serializers.ListField(
        child=serializers.CharField(), required=False, default=list
    )


class RuntimeAgentUpdateSerializer(serializers.Serializer):
    """Every field optional: only the ones sent are changed."""

    name = serializers.CharField(required=False)
    agent = serializers.CharField(required=False, allow_blank=True)
    model = serializers.CharField(required=False)
    system_prompt = serializers.CharField(required=False, allow_blank=True)
    tools = serializers.ListField(child=serializers.CharField(), required=False)
    integrations = serializers.ListField(child=serializers.CharField(), required=False)
    memories = serializers.ListField(child=serializers.CharField(), required=False)
    suggestion_enabled = serializers.BooleanField(required=False)
    title_generation = serializers.BooleanField(required=False)
    max_history = serializers.IntegerField(required=False, allow_null=True)
    file_upload = serializers.BooleanField(required=False)
    vision = serializers.BooleanField(required=False)
    required_tools = serializers.ListField(child=serializers.CharField(), required=False)
    active = serializers.BooleanField(required=False)


class WorkflowDefinitionField(serializers.JSONField):
    """A workflow definition, validated by the SDK's own schema."""

    def to_internal_value(self, data: Any) -> WorkflowDefinition:
        try:
            return WorkflowDefinition.model_validate(super().to_internal_value(data))
        except PydanticValidationError as exc:
            raise serializers.ValidationError(
                [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
            ) from None


class WorkflowCreateSerializer(serializers.Serializer):
    name = serializers.CharField()
    workflow = WorkflowDefinitionField()


class WorkflowUpdateSerializer(serializers.Serializer):
    name = serializers.CharField(required=False)
    workflow = WorkflowDefinitionField(required=False)
    active = serializers.BooleanField(required=False)


class WorkflowRunRequestSerializer(serializers.Serializer):
    workflow = WorkflowDefinitionField()
    inputs = serializers.DictField(required=False, default=dict)


class WorkflowRunByIdSerializer(serializers.Serializer):
    inputs = serializers.DictField(required=False, default=dict)
    run_id = serializers.CharField(required=False, allow_null=True, default=None)


class TraceQuerySerializer(serializers.Serializer):
    message_id = serializers.CharField(required=False, allow_null=True, default=None)
    operation_name = serializers.CharField(required=False, allow_null=True, default=None)


class ReindexQuerySerializer(serializers.Serializer):
    memory_id = serializers.CharField(required=False, allow_null=True, default=None)
    force_rebuild = serializers.BooleanField(required=False, default=False)


# --- Responses --------------------------------------------------------------


class ObjectPermissionsSerializer(serializers.Serializer):
    can_read = serializers.BooleanField()
    can_write = serializers.BooleanField()
    can_manage = serializers.BooleanField()
    can_delete = serializers.BooleanField()


class ThreadSerializer(serializers.Serializer):
    id = serializers.CharField()
    title = serializers.CharField()
    agent_id = serializers.CharField()
    model = serializers.CharField()
    user_id = serializers.CharField(allow_null=True)
    created_at = serializers.DateTimeField()
    updated_at = serializers.DateTimeField()
    metadata = serializers.DictField()
    message_count = serializers.IntegerField()
    file_memory_id = serializers.CharField(allow_null=True)


class ThreadDetailSerializer(serializers.Serializer):
    thread = ThreadSerializer()
    # Messages in the thread's protocol format (e.g. Vercel UI messages), as stored.
    messages = serializers.ListField(child=serializers.DictField())
    permissions = ObjectPermissionsSerializer()


class ThreadFileMetaSerializer(serializers.Serializer):
    file_count = serializers.IntegerField()
    file_memory_id = serializers.CharField(allow_null=True)


class MessageStateSerializer(serializers.Serializer):
    id = serializers.CharField()
    is_deleted = serializers.BooleanField()


class TraceSerializer(serializers.Serializer):
    id = serializers.UUIDField()
    operation_name = serializers.CharField()
    started_at = serializers.DateTimeField()
    ended_at = serializers.DateTimeField(allow_null=True)
    duration_ms = serializers.FloatField(allow_null=True)
    parent_id = serializers.UUIDField(allow_null=True)
    thread_id = serializers.UUIDField(allow_null=True)
    message_id = serializers.UUIDField(allow_null=True)
    agent_id = serializers.UUIDField(allow_null=True)
    agent_name = serializers.CharField()
    model_name = serializers.CharField()
    prompt_tokens = serializers.IntegerField(allow_null=True)
    completion_tokens = serializers.IntegerField(allow_null=True)
    total_tokens = serializers.IntegerField(allow_null=True)
    tags = serializers.DictField()


class TokenUsageSerializer(serializers.Serializer):
    prompt_tokens = serializers.IntegerField()
    completion_tokens = serializers.IntegerField()
    total_tokens = serializers.IntegerField()
    agent_name = serializers.CharField()
    by_subagent = PydanticField()


class AgentSummarySerializer(serializers.Serializer):
    id = serializers.CharField()
    name = serializers.CharField(allow_null=True)
    model = serializers.CharField(allow_null=True)
    file_upload = serializers.BooleanField()
    rag = serializers.BooleanField()
    vision = serializers.BooleanField()


class AgentInfoSerializer(serializers.Serializer):
    id = serializers.CharField()
    name = serializers.CharField(allow_null=True)
    model = serializers.CharField(allow_null=True)
    class_name = serializers.CharField()
    description = serializers.CharField(allow_null=True)
    file_upload = serializers.BooleanField()
    rag = serializers.BooleanField()
    vision = serializers.BooleanField()
    instructions = serializers.CharField(allow_null=True)
    permissions = ObjectPermissionsSerializer()


class ToolSerializer(serializers.Serializer):
    label = serializers.CharField()
    description = serializers.CharField(allow_blank=True)


class AgentIntegrationStatusSerializer(serializers.Serializer):
    server_name = serializers.CharField()
    label = serializers.CharField()
    type = serializers.CharField()
    status = serializers.CharField()
    tool_names = serializers.ListField(child=serializers.CharField())


class AgentToolsSerializer(serializers.Serializer):
    tools = ToolSerializer(many=True)
    integrations = AgentIntegrationStatusSerializer(many=True)


class AgentSettingsSerializer(serializers.ModelSerializer):
    class Meta:
        model = AgentSettings
        fields = [
            "id",
            "name",
            "slug",
            "agent",
            "model",
            "system_prompt",
            "tools",
            "integrations",
            "memories",
            "suggestion_enabled",
            "title_generation",
            "max_history",
            "file_upload",
            "vision",
            "required_tools",
            "active",
            "created_at",
            "updated_at",
        ]


class AgentMemberSerializer(serializers.Serializer):
    """An AgentUser row: who, and whether they manage the agent."""

    user_id = serializers.CharField()
    email = serializers.SerializerMethodField()
    can_manage = serializers.BooleanField()
    created_at = serializers.DateTimeField()

    def get_email(self, obj: Any) -> str:
        return getattr(obj.user, "email", "") or ""


class AgentGroupMemberSerializer(serializers.Serializer):
    group_id = serializers.IntegerField()
    group_name = serializers.CharField(source="group.name")
    can_manage = serializers.BooleanField()
    created_at = serializers.DateTimeField()


class MemoryOutSerializer(serializers.Serializer):
    id = serializers.CharField()
    name = serializers.CharField()
    slug = serializers.CharField()
    description = serializers.CharField(allow_blank=True)
    is_public = serializers.BooleanField()
    document_count = serializers.IntegerField()
    created_at = serializers.CharField()
    updated_at = serializers.CharField()


class MemoryWithPermissionsSerializer(MemoryOutSerializer):
    permissions = ObjectPermissionsSerializer()


class DocumentSerializer(serializers.Serializer):
    id = serializers.CharField()
    file = serializers.CharField(allow_blank=True)
    content = serializers.CharField(allow_blank=True)
    extraction = PydanticField(allow_null=True)
    file_name = serializers.CharField(allow_blank=True)
    data = serializers.DictField()
    file_size = serializers.IntegerField()
    content_type = serializers.CharField(allow_blank=True)
    file_extension = serializers.CharField(allow_blank=True)
    status = serializers.CharField()
    error = serializers.CharField(allow_blank=True)
    error_code = serializers.CharField(allow_blank=True)
    processing_step = serializers.CharField(allow_null=True)
    created_at = serializers.CharField()
    updated_at = serializers.CharField()


class DocumentStatusSerializer(serializers.Serializer):
    id = serializers.CharField()
    status = serializers.CharField()
    error = serializers.CharField(allow_blank=True)
    error_code = serializers.CharField(allow_blank=True)
    processing_step = serializers.CharField(allow_null=True)
    task = PydanticField(allow_null=True)


class UploadResultSerializer(serializers.Serializer):
    id = serializers.CharField()
    status = serializers.CharField()
    processing_step = serializers.CharField(allow_null=True)
    task_id = serializers.CharField(allow_null=True)


class ThreadMemorySerializer(serializers.Serializer):
    id = serializers.CharField()
    name = serializers.CharField()
    description = serializers.CharField(allow_blank=True)
    document_count = serializers.IntegerField()
    active = serializers.BooleanField()
    created_at = serializers.CharField()


class MemoryMemberSerializer(serializers.Serializer):
    user_id = serializers.CharField()
    can_manage = serializers.BooleanField()
    created_at = serializers.CharField()


class MemoryGroupMemberSerializer(serializers.Serializer):
    group_id = serializers.IntegerField()
    group_name = serializers.CharField()
    can_manage = serializers.BooleanField()
    created_at = serializers.CharField()


class UploadSettingsSerializer(serializers.Serializer):
    max_upload_size = serializers.IntegerField()
    allowed_mime_types = serializers.ListField(child=serializers.CharField())


class IntegrationSerializer(serializers.Serializer):
    name = serializers.CharField()
    label = serializers.CharField()
    hint = serializers.CharField(allow_blank=True)
    kind = serializers.CharField()
    status = serializers.CharField()
    supports_connect = serializers.BooleanField()
    supports_test = serializers.BooleanField()
    connect_kind = serializers.CharField(allow_null=True)
    detail = serializers.CharField(allow_null=True)
    connected = serializers.BooleanField(allow_null=True)


class WorkflowSerializer(serializers.ModelSerializer):
    class Meta:
        model = WorkflowSettings
        fields = ["id", "name", "definition", "active"]


class WorkflowRunStepSerializer(serializers.ModelSerializer):
    class Meta:
        model = WorkflowRunStep
        fields = [
            "id",
            "sequence",
            "step_name",
            "output",
            "status",
            "error",
            "started_at",
            "completed_at",
        ]


class WorkflowRunSerializer(serializers.ModelSerializer):
    class Meta:
        model = WorkflowRun
        fields = [
            "id",
            "workflow_id",
            "status",
            "outputs",
            "error",
            "created_at",
            "started_at",
            "completed_at",
        ]


class WorkflowRunDetailSerializer(WorkflowRunSerializer):
    steps = WorkflowRunStepSerializer(many=True, read_only=True)

    class Meta(WorkflowRunSerializer.Meta):
        fields = [*WorkflowRunSerializer.Meta.fields, "steps"]
