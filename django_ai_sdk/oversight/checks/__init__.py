"""System checks for oversight. docs/content/manual/oversight.md lists the ids.

A broken policy fails no request, so these checks are the only place it shows up.
"""

from __future__ import annotations

from typing import Any

PREFIX = "ai_sdk.oversight"


def check_oversight(app_configs: Any = None, **kwargs: Any) -> list[Any]:
    from django_ai_sdk.oversight.checks.agents import agent_checks, policies_in_use
    from django_ai_sdk.oversight.checks.policies import policy_checks
    from django_ai_sdk.oversight.checks.settings import (
        default_policy_checks,
        judge_checks,
    )
    from django_ai_sdk.oversight.conf import setting

    # E012: otherwise a misspelt key is silently ignored and a wrong value misread.
    settings = setting.errors(PREFIX)
    if setting.broken:
        return settings
    in_use = policies_in_use()
    return [
        *policy_checks(in_use),
        *judge_checks(),
        *default_policy_checks(),
        *settings,
        *agent_checks(),
    ]


def check_oversight_deploy(app_configs: Any = None, **kwargs: Any) -> list[Any]:
    """Production-only checks: unmasked content, unmeetable deadlines, no task worker."""
    from django_ai_sdk.oversight.checks.policies import deadline_checks
    from django_ai_sdk.oversight.checks.settings import capture_checks, task_backend_checks
    from django_ai_sdk.oversight.conf import is_enabled, setting

    if setting.broken or not is_enabled():
        return []
    return [*capture_checks(), *deadline_checks(), *task_backend_checks()]


def check_oversight_retention(
    app_configs: Any = None, databases: Any = None, **kwargs: Any
) -> list[Any]:
    """Warn when expired rows are never deleted or open turns never swept. Reads the databases it is given."""
    from django_ai_sdk.oversight.checks.settings import retention_checks, sweep_checks
    from django_ai_sdk.oversight.conf import is_enabled, setting

    if setting.broken or not is_enabled():
        return []
    return [*retention_checks(databases), *sweep_checks(databases)]


__all__ = ["PREFIX", "check_oversight", "check_oversight_deploy", "check_oversight_retention"]
