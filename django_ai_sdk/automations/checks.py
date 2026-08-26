"""Startup diagnostics for automations: a job that never fires produces no error of its own.

Everything here is a Warning, because boot must not fail over a background job.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django.core.checks import Warning as CheckWarning

if TYPE_CHECKING:
    from collections.abc import Sequence

ID_PREFIX = "ai_sdk.automations"


def check_automations(app_configs: Any = None, **kwargs: Any) -> Sequence[Any]:
    """Report anything that would stop a declared automation from running."""
    from django_ai_sdk.automations.registry import get_automations, get_invalid_automations

    refused = get_invalid_automations()
    issues: list[Any] = _check_refused(refused)
    automations = get_automations()
    for name, automation in automations.items():
        issues.extend(_check_schedule(name, automation))
        issues.extend(_check_timezone(name, automation))
        issues.extend(_check_workflow(name, automation))
        issues.extend(_check_requires(name, automation))

    # Refused names count as declared: they already have W006.
    issues.extend(_check_orphaned_config(set(automations) | set(refused)))
    return issues


def _check_refused(refused: dict[str, str]) -> list[Any]:
    """Declarations register() kept out of the registry, one Warning each."""
    return [
        CheckWarning(
            f"Automation {name!r} was not registered: {reason}",
            hint="Fix the declaration in the app's automations module.",
            id=f"{ID_PREFIX}.W006",
        )
        for name, reason in sorted(refused.items())
    ]


def _check_schedule(name: str, automation: Any) -> list[Any]:
    try:
        automation.get_schedule()
    except Exception as exc:
        return [
            CheckWarning(
                f"Automation {name!r} has no usable schedule: {exc}",
                hint="Fix `cron` on the class, or the CRON entry AI_SDK_AUTOMATIONS sets.",
                id=f"{ID_PREFIX}.W001",
            )
        ]
    return []


def _check_timezone(name: str, automation: Any) -> list[Any]:
    """A cron schedule naming a zone that does not exist.

    Otherwise silent: it falls back to UTC and fires an hour off for part of the year.
    """
    from django_ai_sdk.automations.schedule import timezone_available

    if not automation.cron or timezone_available(automation.timezone):
        return []
    return [
        CheckWarning(
            f"Automation {name!r} sets timezone {automation.timezone!r}, which is not a "
            "known IANA zone. Its cron expression is being read as UTC.",
            hint="Use a name like 'Europe/Amsterdam' or 'America/New_York'.",
            id=f"{ID_PREFIX}.W002",
        )
    ]


def _check_workflow(name: str, automation: Any) -> list[Any]:
    """An automation naming a workflow nothing declares.

    Code-declared only: the check is synchronous, so a stored workflow cannot be seen.
    """
    from django_ai_sdk.workflows.registry import get_declared_workflows

    declared = get_declared_workflows()
    if not automation.workflow or automation.workflow in declared:
        return []
    return [
        CheckWarning(
            f"Automation {name!r} runs workflow {automation.workflow!r}, which no "
            "installed app declares. If it is not a database workflow, its runs will "
            "be skipped.",
            hint="Declared workflows: " + (", ".join(sorted(declared)) or "none"),
            id=f"{ID_PREFIX}.W003",
        )
    ]


def _check_requires(name: str, automation: Any) -> list[Any]:
    """Integrations an automation names in `requires` that nothing registers."""
    from django_ai_sdk.integrations.registry import get_declared_integrations

    declared = get_declared_integrations()
    return [
        CheckWarning(
            f"Automation {name!r} requires integration {required!r}, which no installed "
            "app registers.",
            hint=(
                f"Add the app providing {required!r} to INSTALLED_APPS, or drop it from `requires`."
            ),
            id=f"{ID_PREFIX}.W004",
        )
        for required in automation.requires
        if required not in declared
    ]


def _check_orphaned_config(registered: set[str]) -> list[Any]:
    from django_ai_sdk.automations.config import configured_names

    orphaned = configured_names() - registered
    if not orphaned:
        return []
    return [
        CheckWarning(
            "AI_SDK_AUTOMATIONS configures "
            f"{', '.join(sorted(repr(n) for n in orphaned))}, which nothing declares.",
            hint=(
                "Check the spelling, and that the app declaring the automation is in "
                "INSTALLED_APPS. Until then those settings do nothing."
            ),
            id=f"{ID_PREFIX}.W005",
        )
    ]


__all__ = ["check_automations"]
