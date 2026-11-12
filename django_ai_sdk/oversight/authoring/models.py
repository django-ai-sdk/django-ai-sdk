"""Authored policies: one row per policy, edited in place."""

from __future__ import annotations

import uuid

from django.db import models


class AuthoredPolicy(models.Model):
    """A policy an operator defined at runtime.

    `active` False turns its own lenses and rules off: the policies it extends still run.
    Deleting the row unregisters the policy, so turns that name it are refused.
    """

    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True)
    name = models.SlugField(max_length=100, unique=True)
    label = models.CharField(max_length=200, blank=True)
    description = models.TextField(blank=True)
    # Names of parent policies: code policies, packs or other authored policies.
    extends = models.JSONField(default=list, blank=True)
    # Lens and rule dicts in the shape `definitions.build_policy_class` reads.
    lenses = models.JSONField(default=list, blank=True)
    rules = models.JSONField(default=list, blank=True)
    # capture, stream_mode, fail_mode and retention. A missing key is inherited.
    options = models.JSONField(default=dict, blank=True)
    active = models.BooleanField(default=True)
    # A plain id, not a foreign key, so deleting a user keeps the policy.
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


__all__ = ["AuthoredPolicy"]
