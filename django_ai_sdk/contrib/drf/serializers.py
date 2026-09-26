"""Serializers for the rows services hand back as model instances."""

from __future__ import annotations

from typing import Any

from rest_framework import serializers

from django_ai_sdk.agents.models import AgentSettings
from django_ai_sdk.workflows.models import WorkflowRun, WorkflowRunStep, WorkflowSettings


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
            "active",
            "created_at",
            "updated_at",
        ]


class MemberSerializer(serializers.Serializer):
    """An AgentUser row: who, and whether they manage the agent."""

    user_id = serializers.CharField()
    email = serializers.SerializerMethodField()
    can_manage = serializers.BooleanField()
    created_at = serializers.DateTimeField()

    def get_email(self, obj: Any) -> str:
        return getattr(obj.user, "email", "") or ""


class GroupMemberSerializer(serializers.Serializer):
    group_id = serializers.IntegerField()
    group_name = serializers.CharField(source="group.name")
    can_manage = serializers.BooleanField()
    created_at = serializers.DateTimeField()


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
