from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, ClassVar

from django.conf import settings
from django.db import IntegrityError, models, transaction
from django.utils.text import slugify

from django_ai_sdk.workflows.registry import ADHOC, NAME_MAX_LENGTH

if TYPE_CHECKING:
    from django.db.models.fields.related_descriptors import RelatedManager

APP_LABEL = "django_ai_sdk_workflows"

SLUG_ATTEMPTS = 3


class Workflow(models.Model):
    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)
    name = models.CharField(max_length=255)
    slug = models.SlugField(unique=True, max_length=NAME_MAX_LENGTH)
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
        app_label = APP_LABEL
        db_table = "django_ai_sdk_workflows"
        ordering = ["name"]
        verbose_name = "Workflow"
        verbose_name_plural = "Workflows"
        constraints = [
            models.CheckConstraint(
                condition=~models.Q(slug=ADHOC), name="django_ai_sdk_workflow_slug_not_adhoc"
            ),
        ]

    if TYPE_CHECKING:
        created_by_id: Any

    def __str__(self) -> str:
        return self.name

    def save(self, *args: Any, **kwargs: Any) -> None:
        if self.slug:
            self._save_named(*args, **kwargs)
            return
        for attempt in range(SLUG_ATTEMPTS):
            self.slug = self._unique_slug()
            try:
                with transaction.atomic():
                    self._save_named(*args, **kwargs)
                return
            except IntegrityError:
                if attempt == SLUG_ATTEMPTS - 1 or not self.has_slug(self.slug):
                    raise

    def _save_named(self, *args: Any, **kwargs: Any) -> None:
        if isinstance(self.definition, dict) and self.definition.get("name") != self.slug:
            self.definition = {**self.definition, "name": self.slug}
            if kwargs.get("update_fields") is not None:
                kwargs["update_fields"] = {*kwargs["update_fields"], "definition"}
        super().save(*args, **kwargs)

    def has_slug(self, slug: str) -> bool:
        return Workflow.objects.exclude(pk=self.pk).filter(slug=slug).exists()

    def _unique_slug(self) -> str:
        base = (slugify(self.name) or f"workflow-{str(self.pk)[:8]}")[:NAME_MAX_LENGTH]
        slug, suffix = base, 2
        while slug == ADHOC or self.has_slug(slug):
            tail = f"-{suffix}"
            slug = f"{base[: NAME_MAX_LENGTH - len(tail)]}{tail}"
            suffix += 1
        return slug


class WorkflowRunQuerySet(models.QuerySet):
    def held(self, pk: Any, attempt: int) -> WorkflowRunQuerySet:
        return self.filter(pk=pk, attempt=attempt, status=WorkflowRun.Status.RUNNING)


if TYPE_CHECKING:

    class WorkflowRunManager(models.Manager["WorkflowRun"]):
        def get_queryset(self) -> WorkflowRunQuerySet: ...
        def held(self, pk: Any, attempt: int) -> WorkflowRunQuerySet: ...

else:
    WorkflowRunManager = models.Manager.from_queryset(WorkflowRunQuerySet)


class WorkflowRun(models.Model):
    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"
        CANCELLED = "cancelled", "Cancelled"

    workflow = models.ForeignKey(
        Workflow,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="runs",
    )
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.PENDING, db_index=True
    )
    workflow_definition = models.JSONField(null=True, blank=True)
    workflow_key = models.CharField(max_length=NAME_MAX_LENGTH, default=ADHOC, db_index=True)
    trigger = models.CharField(max_length=120, default="manual", db_index=True)

    inputs = models.JSONField(default=dict, blank=True)
    outputs = models.JSONField(null=True, blank=True)
    error = models.TextField(blank=True, default="")
    task_id = models.CharField(max_length=128, null=True, blank=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="workflow_runs",
    )
    started_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="+",
    )
    attempt = models.PositiveIntegerField(default=0)
    heartbeat_at = models.DateTimeField(null=True, blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    objects: ClassVar[WorkflowRunManager] = WorkflowRunManager()

    if TYPE_CHECKING:
        steps: RelatedManager[StepRun]
        user_id: Any
        started_by_id: Any
        workflow_id: Any

    class Meta:
        app_label = APP_LABEL
        db_table = "django_ai_sdk_workflow_runs"
        ordering = ["-created_at"]
        verbose_name = "Workflow Run"
        verbose_name_plural = "Workflow Runs"
        indexes = [
            models.Index(
                fields=["workflow_key", "trigger", "-created_at"],
                name="ai_sdk_wfrun_trigger_latest",
            ),
            models.Index(fields=["workflow", "-created_at"], name="ai_sdk_wfrun_workflow_latest"),
            models.Index(fields=["user", "-created_at"], name="ai_sdk_wfrun_user_latest"),
        ]

    def __str__(self) -> str:
        return f"{self.workflow_key} — {self.status} — {self.created_at}"


class StepRun(models.Model):
    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"
        SKIPPED = "skipped", "Skipped"
        WAITING = "waiting", "Waiting"

    run = models.ForeignKey(WorkflowRun, on_delete=models.CASCADE, related_name="steps")

    if TYPE_CHECKING:
        run_id: Any
    sequence = models.PositiveIntegerField()
    step_name = models.CharField(max_length=255, blank=True, default="")
    output = models.JSONField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    error = models.TextField(blank=True, default="")
    detail = models.CharField(max_length=255, blank=True, default="")
    turn_id = models.CharField(max_length=64, blank=True, default="")
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        app_label = APP_LABEL
        db_table = "django_ai_sdk_workflow_run_steps"
        unique_together = [("run", "sequence")]
        ordering = ["sequence"]
        verbose_name = "Step run"
        verbose_name_plural = "Step runs"
        permissions = [("approve_steprun", "Can decide an approval step")]

    def __str__(self) -> str:
        return f"{self.run_id} step {self.sequence} ({self.step_name})"
