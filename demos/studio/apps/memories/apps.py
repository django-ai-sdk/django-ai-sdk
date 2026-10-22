from __future__ import annotations

from django.apps import AppConfig


class MemoriesConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.memories"
    verbose_name = "Memories & Documents"
