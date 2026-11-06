"""Stored versions of authored policies. The newest version per name is in force."""

from __future__ import annotations

from django.db import models

from django_ai_sdk.oversight.models import OperatorChange


class AuthoredPolicy(OperatorChange):
    """One saved version of a runtime-defined policy.

    Each save adds a row with the next `version`. The newest row per `name` is the one
    in force, unless it is `withdrawn`. Older rows are kept as history.
    """

    name = models.SlugField(max_length=100)
    label = models.CharField(max_length=200, blank=True)
    description = models.TextField(blank=True)
    # Names of parent policies: code policies, packs or other authored policies.
    extends = models.JSONField(default=list, blank=True)
    # Lens and rule dicts in the shape `definitions.build_policy_class` reads.
    lenses = models.JSONField(default=list, blank=True)
    rules = models.JSONField(default=list, blank=True)
    # capture, stream_mode, fail_mode and retention. A missing key is inherited.
    options = models.JSONField(default=dict, blank=True)
    # When the newest version is withdrawn, turns that name the policy are refused.
    withdrawn = models.BooleanField(default=False)

    class Meta:  # ty: ignore[invalid-attribute-override]
        db_table = "django_ai_sdk_oversight_authored_policies"
        ordering = ["name", "-version"]
        constraints = [
            models.UniqueConstraint(fields=["name", "version"], name="oversight_one_policy_version")
        ]
        verbose_name = "authored policy"
        verbose_name_plural = "authored policies"

    def __str__(self) -> str:
        return f"{self.name} v{self.version}{' (withdrawn)' if self.withdrawn else ''}"

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
