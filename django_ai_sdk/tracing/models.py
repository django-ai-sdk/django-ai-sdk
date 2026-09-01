from __future__ import annotations

import uuid
from typing import ClassVar

from django.db import models

from django_ai_sdk.tracing.managers import TraceManager, TraceQuerySet


class Trace(models.Model):
    """A single span, persisted for observability."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    operation_name = models.CharField(max_length=255)
    started_at = models.DateTimeField()
    ended_at = models.DateTimeField(null=True, blank=True)
    duration_ms = models.FloatField(null=True, blank=True)

    parent = models.ForeignKey(
        "self",
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="children",
    )

    thread = models.ForeignKey(
        "django_ai_sdk.Thread",
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="traces",
        db_constraint=False,
    )
    message = models.ForeignKey(
        "django_ai_sdk.Message",
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="traces",
        db_constraint=False,
    )

    # Type hints for FK id attrs
    parent_id: uuid.UUID | None
    thread_id: uuid.UUID | None
    message_id: uuid.UUID | None

    tags = models.JSONField(default=dict, blank=True)
    model_name = models.CharField(max_length=255, blank=True)
    prompt_tokens = models.IntegerField(null=True, blank=True)
    completion_tokens = models.IntegerField(null=True, blank=True)
    total_tokens = models.IntegerField(null=True, blank=True)

    # Typed as the queryset the manager proxies
    objects: ClassVar[TraceQuerySet] = TraceManager()  # ty: ignore[invalid-assignment]

    class Meta:
        app_label = "django_ai_sdk_tracing"
        db_table = "django_ai_sdk_traces"
        indexes = [
            models.Index(fields=["operation_name", "started_at"]),
            models.Index(fields=["thread", "started_at"]),
        ]
        ordering = ["-started_at"]

    def __str__(self) -> str:
        return f"{self.operation_name} ({self.duration_ms or '...'} ms)"
