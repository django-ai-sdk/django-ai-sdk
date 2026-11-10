"""System checks for switches. docs/content/oversight-switches.md lists the ids."""

from __future__ import annotations

from typing import Any

from django.core.checks import Warning

PREFIX = "ai_sdk.oversight.switches"


def check_switches(app_configs: Any = None, **kwargs: Any) -> list[Any]:
    """W001 when runtime agents exist, DEFAULT_POLICY is empty and a base names no policy."""
    from django_ai_sdk.oversight import conf as oversight_conf
    from django_ai_sdk.oversight.checks.agents import agent_policy_name, agent_runtime_bases
    from django_ai_sdk.utils import resolve_setting

    if oversight_conf.setting.broken:
        return []  # Core's E012 reports it.
    if not oversight_conf.is_enabled() or oversight_conf.setting("DEFAULT_POLICY"):
        return []
    if not resolve_setting("AI_SDK_RUNTIME_AGENT_BASES", []):
        return []
    bare = []
    for cls in agent_runtime_bases():
        try:
            if not agent_policy_name(cls):
                bare.append(cls.__qualname__)
        except Exception:
            continue  # E011 reports it.
    if not bare:
        return []
    return [
        Warning(
            f"Runtime agents built on {', '.join(bare)} run unguarded until a switch "
            "assigns them a policy, and any user who may create an agent can make one.",
            hint="Set AI_SDK_OVERSIGHT['DEFAULT_POLICY'], or give the runtime base class "
            "(AI_SDK_RUNTIME_AGENT_BASES) an `oversight` policy.",
            id=f"{PREFIX}.W001",
        )
    ]


__all__ = ["PREFIX", "check_switches"]
