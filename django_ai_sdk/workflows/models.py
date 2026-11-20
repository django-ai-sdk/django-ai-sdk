from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, ClassVar

from django.conf import settings
from django.db import models
from django.utils.text import slugify

from django_ai_sdk.workflows.registry import ADHOC, NAME_MAX_LENGTH

if TYPE_CHECKING:
    from django.db.models import Manager

APP_LABEL = "django_ai_sdk_workflows"


class Workflow(models.Model):
    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)
    name = models.CharField(max_length=255)
    # Registry key shared with code-declared workflows.
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
            # "adhoc" is the key for runs of unstored, unregistered definitions. A
            # stored workflow with that slug would share their channel and rules.
            models.CheckConstraint(
                condition=~models.Q(slug=ADHOC), name="django_ai_sdk_workflow_slug_not_adhoc"
            ),
        ]

    if TYPE_CHECKING:
        created_by_id: Any

    def __str__(self) -> str:
        return self.name

    def save(self, *args: Any, **kwargs: Any) -> None:
        if not self.slug:
            self.slug = self._unique_slug()
        # The definition's name is always the slug, so a row cannot claim another's name.
        if isinstance(self.definition, dict) and self.definition.get("name") != self.slug:
            self.definition = {**self.definition, "name": self.slug}
            if kwargs.get("update_fields") is not None:
                kwargs["update_fields"] = {*kwargs["update_fields"], "definition"}
        super().save(*args, **kwargs)

    def _unique_slug(self) -> str:
        """A slug derived from `name`, suffixed until it is free."""
        base = (slugify(self.name) or f"workflow-{str(self.pk)[:8]}")[:NAME_MAX_LENGTH]
        slug, suffix = base, 2
        taken = Workflow.objects.exclude(pk=self.pk)
        # A row may share a registered workflow's name, e.g. to show it in a studio:
        # the registered one wins at start, and the row is still reached by id.
        while slug == ADHOC or taken.filter(slug=slug).exists():
            tail = f"-{suffix}"
            slug = f"{base[: NAME_MAX_LENGTH - len(tail)]}{tail}"
            suffix += 1
        return slug


class WorkflowRunQuerySet(models.QuerySet):
    def held(self, pk: Any, attempt: int) -> WorkflowRunQuerySet:
        """The run, if `attempt` still holds it: the latest claim, still running."""
        return self.filter(pk=pk, attempt=attempt, status=WorkflowRun.Status.RUNNING)


class WorkflowRun(models.Model):
    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"
        # Stopped by a person (`WorkflowService.cancel`).
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
    # Set by the server when the run opens, never by the caller. It is the stored
    # workflow's slug, the registered workflow's name, or "adhoc". Step turns run
    # on the channel "workflow:<workflow_key>:<step>".
    workflow_key = models.CharField(max_length=NAME_MAX_LENGTH, default=ADHOC, db_index=True)
    # What started the run: "manual" (a person or code), "cron:<trigger>" or
    # "event:<trigger>" (`workflows.triggers`).
    trigger = models.CharField(max_length=120, default="manual", db_index=True)

    inputs = models.JSONField(default=dict, blank=True)
    outputs = models.JSONField(null=True, blank=True)
    error = models.TextField(blank=True, default="")
    # The DBOS workflow id of the current attempt (`workflows.durable.workflow_id`).
    task_id = models.CharField(max_length=128, null=True, blank=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="workflow_runs",
    )
    # The person who asked for the run, when one did: `user` for a manual run, the
    # operator who fired a trigger by hand. Like `user`, they cannot approve it.
    started_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="+",
    )
    # How many times a worker has claimed this run. A worker only writes while its
    # attempt is still the latest.
    attempt = models.PositiveIntegerField(default=0)
    # Refreshed at the claim and at every step: when the run last made progress.
    heartbeat_at = models.DateTimeField(null=True, blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    objects: ClassVar[WorkflowRunQuerySet] = WorkflowRunQuerySet.as_manager()  # ty: ignore[invalid-assignment]

    if TYPE_CHECKING:
        steps: Manager[StepRun]
        # Django adds the FK's `_id` attribute at class build time, which a static
        # checker reading this module does not see.
        user_id: Any
        started_by_id: Any
        workflow_id: Any

    class Meta:
        app_label = APP_LABEL
        db_table = "django_ai_sdk_workflow_runs"
        ordering = ["-created_at"]
        verbose_name = "Workflow Run"
        verbose_name_plural = "Workflow Runs"

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
        # An approval step waiting for a person's decision.
        WAITING = "waiting", "Waiting"

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
    # The oversight turn the step's model call opened, if it opened one.
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
