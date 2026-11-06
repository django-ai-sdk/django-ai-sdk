"""Read the switches in force from the database, cached per process.

Resolved policies and agent assignments are cached for RUNTIME_CACHE_SECONDS, so
other processes see a switch within that time. The process that made the switch
clears its cache at once. If the database cannot be read, turns use the last read,
or code alone.
"""

from __future__ import annotations

from typing import Any

from asgiref.sync import sync_to_async

from django_ai_sdk.oversight.conf import setting
from django_ai_sdk.oversight.resolve import OverlaySource


def in_force(**filters: Any) -> Any:
    """The newest switch of each rule or agent, unless it hands back to code."""
    from django_ai_sdk.oversight.switches.models import Override

    newest = Override.objects.filter(**filters).newest("kind", "policy", "target")
    return newest.exclude(value=None)


def rule_modes(names: list[str]) -> dict[str, dict[str, str]]:
    """{policy: {rule: mode}} for the switched rules of the policies `names`."""
    from django_ai_sdk.oversight.switches.models import Override

    found: dict[str, dict[str, str]] = {name: {} for name in names}
    rows = in_force(kind=Override.Kind.RULE_MODE, policy__in=names)
    for policy, rule, mode in rows.values_list("policy", "target", "value"):
        found[policy][rule] = mode
    return found


class DatabaseOverlay(OverlaySource):
    """Overlay source backed by the `Override` rows in force."""

    @property
    def ttl(self) -> float:  # ty: ignore[invalid-property-type-override]
        return float(setting("RUNTIME_CACHE_SECONDS"))

    async def rule_modes(self, names: list[str]) -> dict[str, dict[str, str]]:
        return await sync_to_async(rule_modes)(names)

    async def agent_policies(self) -> dict[str, str]:
        from django_ai_sdk.oversight.switches.models import Override

        rows = in_force(kind=Override.Kind.AGENT_POLICY).values_list("target", "value")
        return {agent_id: policy async for agent_id, policy in rows}


__all__ = ["DatabaseOverlay", "in_force", "rule_modes"]
