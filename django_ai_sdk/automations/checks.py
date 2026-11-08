"""Startup diagnostics for automations.

Warnings only, because a background job must never stop the site from booting.
"""

from __future__ import annotations

from typing import Any

from django.core.checks import Warning as CheckWarning

PREFIX = "ai_sdk.automations"


def check_automations(app_configs: Any = None, **kwargs: Any) -> list[Any]:
    """Report anything that would stop a declared automation from running."""
    from django_ai_sdk.automations.registry import get_automations, get_invalid_automations

    issues = _check_refused(get_invalid_automations())
    for name, automation in get_automations().items():
        issues.extend(_check_workflow(name, automation))
    return issues


def _check_refused(refused: dict[str, str]) -> list[Any]:
    """One warning per declaration that register_automation() rejected."""
    return [
        CheckWarning(
            f"Automation {name!r} was not registered: {reason}",
            hint="Fix the declaration in the app's automations module.",
            id=f"{PREFIX}.W006",
        )
        for name, reason in sorted(refused.items())
    ]


def _check_workflow(name: str, automation: Any) -> list[Any]:
    """Warn when an automation names a workflow that no app declares.

    The check is synchronous, so it cannot see workflows stored in the database.
    """
    from django_ai_sdk.workflows.registry import get_declared_workflows

    declared = get_declared_workflows()
    if automation.workflow in declared:
        return []
    return [
        CheckWarning(
            f"Automation {name!r} runs workflow {automation.workflow!r}, which no "
            "installed app declares. If it is not a database workflow, its runs will "
            "be skipped.",
            hint="Declared workflows: " + (", ".join(sorted(declared)) or "none"),
            id=f"{PREFIX}.W003",
        )
    ]


__all__ = ["check_automations"]
