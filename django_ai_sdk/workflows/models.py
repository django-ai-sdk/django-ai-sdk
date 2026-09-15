from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from django.conf import settings
from django.db import models
from django.utils.text import slugify

if TYPE_CHECKING:
    from django.db.models import Manager

    from django_ai_sdk.workflows.schemas import WorkflowDefinition


class WorkflowSettings(models.Model):
    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)
    name = models.CharField(max_length=255)
    # Registry key shared with code-declared workflows.
    slug = models.SlugField(unique=True, max_length=100)
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

    def save(self, *args: Any, **kwargs: Any) -> None:
        if not self.slug:
            self.slug = self._unique_slug()
        super().save(*args, **kwargs)

    def _unique_slug(self) -> str:
        """A slug derived from `name`, suffixed until it is free."""
        base = (slugify(self.name) or f"workflow-{str(self.pk)[:8]}")[:100]
        slug, suffix = base, 2
        taken = WorkflowSettings.objects.exclude(pk=self.pk)
        while taken.filter(slug=slug).exists():
            tail = f"-{suffix}"
            slug = f"{base[: 100 - len(tail)]}{tail}"
            suffix += 1
        return slug

    def to_workflow_definition(self) -> WorkflowDefinition:
        from django_ai_sdk.workflows.schemas import WorkflowDefinition

        return WorkflowDefinition.model_validate(self.definition)


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

    inputs = models.JSONField(default=dict, blank=True)
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
        # Django adds the FK's `_id` attribute at class build time, which a static
        # checker reading this module does not see.
        user_id: Any
        workflow_id: Any

    class Meta:
        app_label = "django_ai_sdk"
        db_table = "django_ai_sdk_workflow_runs"
        ordering = ["-created_at"]
        verbose_name = "Workflow Run"
        verbose_name_plural = "Workflow Runs"

    def __str__(self) -> str:
        return f"{self.workflow_id or 'inline'} — {self.status} — {self.created_at}"


class WorkflowRunStep(models.Model):
    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"
        SKIPPED = "skipped", "Skipped"

    run = models.ForeignKey(WorkflowRun, on_delete=models.CASCADE, related_name="steps")

    if TYPE_CHECKING:
        run_id: Any
    sequence = models.PositiveIntegerField()
    step_name = models.CharField(max_length=255, blank=True, default="")
    output = models.JSONField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    error = models.TextField(blank=True, default="")
    # One line on the outcome: why a step was skipped, or how it failed.
    detail = models.CharField(max_length=255, blank=True, default="")
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
        return f"{self.run_id} step {self.sequence} ({self.step_name})"
