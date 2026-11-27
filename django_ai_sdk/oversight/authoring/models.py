from __future__ import annotations

import uuid

from django.db import models


class AuthoredPolicy(models.Model):
    """`active` False turns its own lenses and rules off; the policies it extends still run."""

    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True)
    name = models.SlugField(max_length=100, unique=True)
    label = models.CharField(max_length=200, blank=True)
    description = models.TextField(blank=True)
    extends = models.JSONField(default=list, blank=True)
    lenses = models.JSONField(default=list, blank=True)
    rules = models.JSONField(default=list, blank=True)
    options = models.JSONField(default=dict, blank=True)
    active = models.BooleanField(default=True)
    updated_by_id = models.CharField(max_length=64, blank=True)
    updated_at = models.DateTimeField(auto_now=True, db_index=True)

    class Meta:
        db_table = "django_ai_sdk_oversight_authored_policies"
        ordering = ["name"]
        verbose_name = "authored policy"
        verbose_name_plural = "authored policies"

    def __str__(self) -> str:
        return f"{self.name}{'' if self.active else ' (off)'}"

    def definition(self) -> dict:
        return {
            "label": self.label,
            "description": self.description,
            "extends": list(self.extends or []),
            "lenses": list(self.lenses or []),
            "rules": list(self.rules or []),
            "options": dict(self.options or {}),
        }
