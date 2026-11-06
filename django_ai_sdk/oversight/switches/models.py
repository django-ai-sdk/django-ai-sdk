"""Stored switches. The newest version per rule or agent is in force."""

from __future__ import annotations

from django.db import models

from django_ai_sdk.oversight.models import OperatorChange


class Override(OperatorChange):
    """One runtime switch of a rule's mode or an agent's policy.

    Each switch of a rule or agent adds a version. A version with no `value` hands
    control back to code.
    """

    class Kind(models.TextChoices):
        RULE_MODE = "rule_mode"
        AGENT_POLICY = "agent_policy"

    kind = models.CharField(max_length=16, choices=Kind.choices)
    # rule_mode: the policy the switch applies to (child policies inherit it). Else "".
    policy = models.CharField(max_length=100, blank=True)
    # The rule name for rule_mode, the agent id for agent_policy.
    target = models.CharField(max_length=100)
    # A mode for rule_mode. A policy name for agent_policy, or "" for off. None: code decides.
    value = models.CharField(max_length=100, blank=True, null=True)  # noqa: DJ001

    class Meta:  # ty: ignore[invalid-attribute-override]
        db_table = "django_ai_sdk_oversight_overrides"
        ordering = ["-changed_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["kind", "policy", "target", "version"], name="oversight_one_switch_version"
            )
        ]

    def __str__(self) -> str:
        where = f"{self.policy}:{self.target}" if self.policy else self.target
        return f"{self.kind} {where} v{self.version} = {self.value!r}"


__all__ = ["Override"]
