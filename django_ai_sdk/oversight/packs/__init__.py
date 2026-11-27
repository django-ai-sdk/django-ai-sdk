from __future__ import annotations

from django_ai_sdk.oversight.packs import quality, safety, security  # noqa: F401
from django_ai_sdk.oversight.policy import Policy
from django_ai_sdk.oversight.registry import register_policy


@register_policy
class DefaultPolicy(Policy):
    """All packs with every rule in report mode."""

    name = "default"
    label = "Default"
    description = "All built-in packs, every rule in report mode."
    origin = "pack:default"
    extends = ["security", "safety", "quality"]  # noqa: RUF012
