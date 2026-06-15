import uuid
from collections.abc import Iterable

from django.conf import settings
from django.db import models
from django.db.models.base import ModelBase
from django.utils.text import slugify


class WebAssistantSettings(models.Model):
    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)
    name = models.CharField(max_length=255)
    slug = models.SlugField(unique=True, max_length=100)
    model = models.CharField(max_length=100, default="gpt-4o")
    system_prompt = models.TextField(blank=True, default="")
    base_class = models.CharField(max_length=255, blank=True, default="")
    tools = models.JSONField(default=list, blank=True)
    mcp_servers = models.JSONField(default=list, blank=True)
    suggestion_enabled = models.BooleanField(default=False)
    title_generation = models.BooleanField(default=True)
    max_history = models.PositiveIntegerField(null=True, blank=True)
    file_upload = models.BooleanField(default=False)
    active = models.BooleanField(default=True, db_index=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="web_assistants",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = "django_ai_sdk"
        db_table = "django_ai_sdk_web_assistants"
        ordering = ["name"]
        verbose_name = "Web Assistant"
        verbose_name_plural = "Web Assistants"

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
