from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from django.conf import settings
from django.db import models
from django.utils.text import slugify

if TYPE_CHECKING:
    from collections.abc import Iterable

    from django.db.models.base import ModelBase


class AssistantSettings(models.Model):
    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)
    name = models.CharField(max_length=255)
    slug = models.SlugField(unique=True, max_length=100)
    model = models.CharField(max_length=255)
    system_prompt = models.TextField(blank=True, default="")
    assistant = models.CharField(max_length=255, blank=True, default="")
    tools = models.JSONField(default=list, blank=True)
    memories = models.JSONField(default=list, blank=True)
    integrations = models.JSONField(default=dict, blank=True)
    suggestion_enabled = models.BooleanField(default=False)
    title_generation = models.BooleanField(default=True)
    max_history = models.PositiveIntegerField(null=True, blank=True)
    file_upload = models.BooleanField(default=False)
    active = models.BooleanField(default=True, db_index=True)

    # Reverse relation type hints
    assistant_users: models.Manager[AssistantUser]
    assistant_groups: models.Manager[AssistantGroup]

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = "django_ai_sdk"
        db_table = "django_ai_sdk_runtime_assistants"
        ordering = ["name"]
        verbose_name = "Assistant Settings"
        verbose_name_plural = "Assistant Settings"

    def __str__(self) -> str:
        return self.name

    def save(
        self,
        *,
        force_insert: bool | tuple[ModelBase, ...] = False,
        force_update: bool = False,
        using: str | None = None,
        update_fields: Iterable[str] | None = None,
    ) -> None:
        if not self.slug:
            self.slug = slugify(self.name)
        super().save(
            force_insert=force_insert,
            force_update=force_update,
            using=using,
            update_fields=update_fields,
        )


class AssistantUser(models.Model):
    assistant = models.ForeignKey(
        AssistantSettings, on_delete=models.CASCADE, related_name="assistant_users"
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="assistants",
    )
    can_manage = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    assistant_id: int
    user_id: int

    class Meta:
        app_label = "django_ai_sdk"
        db_table = "django_ai_sdk_runtime_assistant_users"
        unique_together = [["assistant", "user"]]

    def __str__(self) -> str:
        return f"{self.user} - {self.assistant.name}"


class AssistantGroup(models.Model):
    assistant = models.ForeignKey(
        AssistantSettings,
        on_delete=models.CASCADE,
        related_name="assistant_groups",
    )
    group = models.ForeignKey(
        "auth.Group",
        on_delete=models.CASCADE,
        related_name="assistant_group_links",
    )
    can_manage = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    assistant_id: int
    group_id: int

    class Meta:
        app_label = "django_ai_sdk"
        db_table = "django_ai_sdk_runtime_assistant_groups"
        unique_together = [["assistant", "group"]]

    def __str__(self) -> str:
        return f"{self.group} - {self.assistant.name}"
