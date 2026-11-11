"""Checks on AI_SDK_OVERSIGHT and the deployment.

Covers the settings schema, judge aliases, the default policy, plain content, the
task backend and pruning.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from django.core.checks import Error, Warning

from django_ai_sdk.oversight.checks import PREFIX


def judge_checks() -> list[Any]:
    """Check that every judge alias builds (E005) and can run here (W005)."""
    from django_ai_sdk.oversight.conf import setting
    from django_ai_sdk.oversight.judges.base import resolve

    messages: list[Any] = []
    for alias in setting("JUDGES") or {}:
        try:
            judge = resolve(alias)
        except Exception as exc:
            messages.append(
                Error(f"Judge alias {alias!r} cannot be built: {exc}", id=f"{PREFIX}.E005")
            )
            continue
        if missing := judge.missing():
            messages.append(
                Warning(
                    f"Judge alias {alias!r} is {type(judge).__name__}, but {missing}: it abstains.",
                    id=f"{PREFIX}.W005",
                )
            )
    return messages


def default_policy_checks() -> list[Any]:
    """Check that DEFAULT_POLICY names a registered policy (E006)."""
    from django_ai_sdk.oversight.conf import setting
    from django_ai_sdk.oversight.registry import get_policies

    default = setting("DEFAULT_POLICY")
    if default and default not in get_policies():
        return [
            Error(
                f"AI_SDK_OVERSIGHT['DEFAULT_POLICY'] names unknown policy {default!r}",
                id=f"{PREFIX}.E006",
            )
        ]
    return []


def capture_checks() -> list[Any]:
    """Warn when CAPTURE is "full", so the stored content is unmasked."""
    from django_ai_sdk.oversight.conf import setting

    if str(setting("CAPTURE")) != "full":
        return []
    return [
        Warning(
            "Oversight keeps every question and answer unmasked: "
            "AI_SDK_OVERSIGHT['CAPTURE'] is 'full'.",
            hint="Use CAPTURE 'masked' (the default).",
            id=f"{PREFIX}.W001",
        )
    ]


def task_backend_checks() -> list[Any]:
    """Warn when the task backend runs reflect inline, or never."""
    from django_ai_sdk.checks import TASKS_HINT, task_backend_problem
    from django_ai_sdk.oversight.tasks import reflect_turn

    if problem := task_backend_problem(reflect_turn):
        return [
            Warning(
                f"Oversight's reflect stage and notifications {problem}.",
                hint=TASKS_HINT,
                id=f"{PREFIX}.W014",
            )
        ]
    return []


# How long `oversight_prune` may go without running before W007.
PRUNE_DUE = timedelta(days=2)


def retention_checks(databases: Any) -> list[Any]:
    """Warn when rows expired more than PRUNE_DUE ago are still there: nothing prunes.

    Retention only stamps `expires_at`. Runs under `manage.py check --deploy --database <alias>`.
    """
    from django.utils import timezone

    from django_ai_sdk.checks import per_database
    from django_ai_sdk.oversight.models import Turn, TurnContent

    due = timezone.now() - PRUNE_DUE

    def overdue(alias: str) -> bool:
        return (
            Turn.objects.using(alias).filter(expires_at__lt=due).exists()
            or TurnContent.objects.using(alias).filter(expires_at__lt=due).exists()
        )

    return [
        Warning(
            f"Expired oversight rows in database {alias!r} are not deleted: "
            "oversight_prune does not run.",
            hint="Schedule `manage.py oversight_prune` daily; RETENTION only stamps "
            "expires_at on each row.",
            id=f"{PREFIX}.W007",
        )
        for alias, late in per_database(databases, overdue)
        if late
    ]
