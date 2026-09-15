"""Persistence for the scheduler: one cursor row per automation, one row per run."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from django.conf import settings
from django.db import models

if TYPE_CHECKING:
    from django.db.models import Manager


class AutomationState(models.Model):
    """The scheduler's cursor and lease for one automation.

    Rows are created lazily by the first tick, never at app load.
    """

    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)
    name = models.CharField(max_length=255, unique=True, db_index=True)

    # Operator override. Null means "no opinion, defer to settings then the class".
    enabled = models.BooleanField(null=True, blank=True)

    # Advanced by the claim itself, so a winning tick has already scheduled the next.
    next_run_at = models.DateTimeField(db_index=True)

    last_dispatched_at = models.DateTimeField(null=True, blank=True)

    # Feeds a prompt's {last_run_at}. Only advanced on success, so a failed run does
    # not skip the window it never processed.
    last_success_at = models.DateTimeField(null=True, blank=True)

    # Held while a run is in flight. Expiry is the crash-recovery path.
    locked_until = models.DateTimeField(null=True, blank=True, db_index=True)

    schedule_repr = models.CharField(max_length=255, blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    if TYPE_CHECKING:
        runs: Manager[AutomationRun]

    class Meta:
        app_label = "django_ai_sdk"
        db_table = "django_ai_sdk_automation_state"
        ordering = ["name"]
        verbose_name = "Automation"
        verbose_name_plural = "Automations"

    def __str__(self) -> str:
        return self.name


class AutomationRun(models.Model):
    """One execution of one automation, for one principal."""

    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        SUCCEEDED = "succeeded", "Succeeded"
        FAILED = "failed", "Failed"
        # Did not run, and that is not an error.
        SKIPPED = "skipped", "Skipped"

    class Trigger(models.TextChoices):
        SCHEDULE = "schedule", "Schedule"
        MANUAL = "manual", "Manual"

    # Denormalised from state, so deleting the code keeps the audit trail.
    name = models.CharField(max_length=255, db_index=True)
    state = models.ForeignKey(
        AutomationState,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="runs",
    )

    # Shared by every run one tick produced.
    dispatch_id = models.UUIDField(default=uuid.uuid4, db_index=True)

    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.PENDING, db_index=True
    )
    trigger = models.CharField(max_length=20, choices=Trigger.choices, default=Trigger.SCHEDULE)

    # Null for an app-level run.
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="automation_runs",
    )

    if TYPE_CHECKING:
        user_id: Any
        workflow_run_id: Any

    workflow_run = models.ForeignKey(
        "django_ai_sdk.WorkflowRun",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="automation_runs",
    )

    # The window this run is for, not when it started.
    scheduled_for = models.DateTimeField(db_index=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    output = models.JSONField(null=True, blank=True)
    error = models.TextField(blank=True, default="")
    skip_reason = models.CharField(max_length=255, blank=True, default="")

    # Correlates with the worker's own record.
    task_id = models.CharField(max_length=64, null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = "django_ai_sdk"
        db_table = "django_ai_sdk_automation_runs"
        ordering = ["-scheduled_for", "-created_at"]
        verbose_name = "Automation Run"
        verbose_name_plural = "Automation Runs"
        indexes = [
            models.Index(fields=["name", "-scheduled_for"]),
        ]

    def __str__(self) -> str:
        return f"{self.name} — {self.status} — {self.scheduled_for:%Y-%m-%d %H:%M}"


class AutomationSubscription(models.Model):
    """One user's opt-in to one automation.

    Denormalised on `name`, so it survives the automation leaving INSTALLED_APPS.
    """

    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)
    name = models.CharField(max_length=255, db_index=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="automation_subscriptions",
    )
    enabled = models.BooleanField(default=False)

    if TYPE_CHECKING:
        user_id: Any

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = "django_ai_sdk"
        db_table = "django_ai_sdk_automation_subscriptions"
        constraints = [
            models.UniqueConstraint(fields=["name", "user"], name="unique_automation_subscriber")
        ]

    def __str__(self) -> str:
        return f"{self.name} — {self.user_id} — {'on' if self.enabled else 'off'}"
