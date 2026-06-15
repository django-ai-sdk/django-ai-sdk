from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from django.conf import settings
from django.db import models

if TYPE_CHECKING:
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
        from django_ai_sdk.workflows.schemas import WorkflowDefinition as WD

        return WD.model_validate(self.definition)

    @classmethod
    def from_workflow_definition(
        cls, name: str, workflow: WorkflowDefinition, **kwargs: Any
    ) -> WorkflowSettings:
        return cls(name=name, definition=workflow.model_dump(), **kwargs)
