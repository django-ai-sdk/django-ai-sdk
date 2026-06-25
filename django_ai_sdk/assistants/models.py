import uuid
from collections.abc import Iterable

from django.conf import settings
from django.db import models
from django.db.models.base import ModelBase
from django.utils.text import slugify


class AssistantSettings(models.Model):
    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)
    name = models.CharField(max_length=255)
    slug = models.SlugField(unique=True, max_length=100)
    model = models.CharField(max_length=100, default="gpt-4o")
    system_prompt = models.TextField(blank=True, default="")
    assistant = models.CharField(max_length=255, blank=True, default="")
    tools = models.JSONField(default=list, blank=True)
    mcp_servers = models.JSONField(default=list, blank=True)
    # Knowledge bases linked to threads created with this assistant (Memory slugs).
    memories = models.JSONField(default=list, blank=True)
    # Flat access-level permission keys (resolved via AI_SDK_RUNTIME_ASSISTANT_PERMISSIONS).
    permissions = models.JSONField(default=list, blank=True)
    # Optional access restriction to specific auth Group names (empty = no restriction).
    allowed_groups = models.JSONField(default=list, blank=True)
    suggestion_enabled = models.BooleanField(default=False)
    title_generation = models.BooleanField(default=True)
    max_history = models.PositiveIntegerField(null=True, blank=True)
    file_upload = models.BooleanField(default=False)
    active = models.BooleanField(default=True, db_index=True)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="assistant_settings",
    )
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
