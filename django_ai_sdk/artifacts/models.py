from __future__ import annotations

import uuid

from django.conf import settings
from django.db import models
from django.utils import timezone

from django_ai_sdk.conversation.models import Thread


class Artifact(models.Model):
    """
    Structured artifact submitted by the agent via tool call.
    Linked to a thread; the tool result JSON carries the artifact_id for
    message-level association on the frontend.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    thread = models.ForeignKey(Thread, on_delete=models.CASCADE, related_name="artifacts")
    schema_name = models.CharField(max_length=255)
    artifact_type = models.CharField(max_length=50, default="data_table")
    data = models.JSONField()
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="artifacts",
    )
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    thread_id: str
    created_by_id: str | None

    class Meta:
        app_label = "django_ai_sdk"
        db_table = "django_ai_sdk_artifacts"
        ordering = ["created_at"]

    def __str__(self) -> str:
        return f"{self.schema_name} ({self.artifact_type}) in thread {self.thread_id}"
