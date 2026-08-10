from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from django.conf import settings
from django.db import models

if TYPE_CHECKING:
    from django.db.models import Manager

    from django_ai_sdk.workflows.schemas import WorkflowDefinition


class WorkflowSettings(models.Model):
    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)
    name = models.CharField(max_length=255)
    definition = models.JSONField(default=dict)
    active = models.BooleanField(default=True, db_index=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="workflows",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = "django_ai_sdk"
        db_table = "django_ai_sdk_workflows"
        ordering = ["name"]
        verbose_name = "Workflow"
        verbose_name_plural = "Workflows"

    def __str__(self) -> str:
        return self.name

    def to_workflow_definition(self) -> WorkflowDefinition:
        from django_ai_sdk.workflows.schemas import WorkflowDefinition

        return WorkflowDefinition.model_validate(self.definition)

    @classmethod
    def from_workflow_definition(
        cls, name: str, workflow: WorkflowDefinition, **kwargs: Any
    ) -> WorkflowSettings:
        return cls(name=name, definition=workflow.model_dump(), **kwargs)


class WorkflowRun(models.Model):
    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    workflow = models.ForeignKey(
        WorkflowSettings,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="runs",
    )
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.PENDING, db_index=True
    )
    workflow_definition = models.JSONField(null=True, blank=True)
    input_messages = models.JSONField(default=list)
    outputs = models.JSONField(null=True, blank=True)
    error = models.TextField(blank=True, default="")
    task_id = models.CharField(max_length=64, null=True, blank=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="workflow_runs",
    )
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    if TYPE_CHECKING:
        steps: Manager[WorkflowRunStep]

    class Meta:
        app_label = "django_ai_sdk"
        db_table = "django_ai_sdk_workflow_runs"
        ordering = ["-created_at"]
        verbose_name = "Workflow Run"
        verbose_name_plural = "Workflow Runs"

    def __str__(self) -> str:
        workflow_id = str(getattr(self, "workflow_id", None) or "inline")
        return f"{workflow_id} — {self.status} — {self.created_at}"


class WorkflowRunStep(models.Model):
    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    run = models.ForeignKey(WorkflowRun, on_delete=models.CASCADE, related_name="steps")
    sequence = models.PositiveIntegerField()
    step_name = models.CharField(max_length=255, blank=True, default="")
    output_key = models.CharField(max_length=255)
    output = models.JSONField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    error = models.TextField(blank=True, default="")
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        app_label = "django_ai_sdk"
        db_table = "django_ai_sdk_workflow_run_steps"
        unique_together = [("run", "sequence")]
        ordering = ["sequence"]
        verbose_name = "Workflow Run Step"
        verbose_name_plural = "Workflow Run Steps"

    def __str__(self) -> str:
        run_id = getattr(self, "run_id", "unknown")
        return f"{run_id} step {self.sequence} ({self.output_key})"
