"""Built-in policies: security, quality, safety, and a default that combines them.

Extend a whole pack, or take single lenses and rules with `pick`.
OversightConfig.ready() imports this module, which registers the packs.
"""

from __future__ import annotations

# Imported to register the packs, which DefaultPolicy extends.
from django_ai_sdk.oversight.packs import quality, safety, security  # noqa: F401
from django_ai_sdk.oversight.policy import Policy
from django_ai_sdk.oversight.registry import register_policy


@register_policy
class DefaultPolicy(Policy):
    """All packs with every rule in report mode, so you can observe before enforcing."""

    name = "default"
    label = "Default"
    description = "All built-in packs, every rule in report mode."
    origin = "pack:default"
    extends = ["security", "safety", "quality"]  # noqa: RUF012
