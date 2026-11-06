"""Scheduler tables: one cursor row per automation, and the runs it opened."""

from __future__ import annotations

import uuid

from django.db import models


class AutomationState(models.Model):
    """The scheduler's cursor for one automation.

    Rows are created lazily by the first tick, never at app load.
    """

    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)
    name = models.CharField(max_length=255, unique=True, db_index=True)

    # The claim advances this, so the winning tick has already scheduled the next run.
    next_run_at = models.DateTimeField(null=True, blank=True, db_index=True)

    last_dispatched_at = models.DateTimeField(null=True, blank=True)

    # Operator override of the class's `enabled`. Null lets the class decide.
    enabled = models.BooleanField(null=True, blank=True, default=None)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "django_ai_sdk_automation_state"
        ordering = ["name"]
        verbose_name = "Automation"
        verbose_name_plural = "Automations"

    def __str__(self) -> str:
        return self.name


class AutomationRun(models.Model):
    """Which automation opened a workflow run. The run itself is the workflows app's."""

    run = models.OneToOneField(
        "django_ai_sdk_workflows.WorkflowRun",
        on_delete=models.CASCADE,
        primary_key=True,
        related_name="automation_run",
    )
    # The automation's name. Renaming the automation orphans these rows.
    automation = models.CharField(max_length=255, db_index=True)

    class Meta:
        db_table = "django_ai_sdk_automation_runs"
        verbose_name = "Automation run"

    def __str__(self) -> str:
        return f"{self.automation}: {self.pk}"


__all__ = ["AutomationRun", "AutomationState"]
