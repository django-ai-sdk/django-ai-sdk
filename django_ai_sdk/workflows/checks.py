"""System checks for workflows. docs/content/manual/workflows.md lists the ids."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django.core.checks import Error

if TYPE_CHECKING:
    from collections.abc import Callable

PREFIX = "ai_sdk.workflows"


def check_workflows(app_configs: Any = None, **kwargs: Any) -> list[Any]:
    """Report every declaration and every registry entry that cannot be used."""
    from django.conf import settings as django_settings

    from django_ai_sdk.workflows.conf import load_classes, setting
    from django_ai_sdk.workflows.listeners import RunListener
    from django_ai_sdk.workflows.registry import get_invalid_workflows
    from django_ai_sdk.workflows.schemas import AGENT_STEP, APPROVAL_STEP
    from django_ai_sdk.workflows.steps import Step

    errors: list[Any] = [
        Error(
            reason,
            hint="Fix the declaration in the app's workflows module.",
            id=f"{PREFIX}.E001",
        )
        for _, reason in sorted(get_invalid_workflows().items())
    ]
    errors.extend(
        Error(
            f"{old} is no longer read: workflows now read AI_SDK_WORKFLOWS[{new!r}].",
            hint=f"Move it to AI_SDK_WORKFLOWS = {{{new!r}: ...}}.",
            id=f"{PREFIX}.E005",
        )
        for old, new in (
            ("AI_SDK_WORKFLOW_STEPS", "STEPS"),
            ("AI_SDK_WORKFLOW_ACTIONS", "LISTENERS"),
        )
        if hasattr(django_settings, old)
    )
    settings = settings_errors()
    if settings:
        # The registries below read the same setting.
        return [*errors, *settings]
    for reserved in sorted({AGENT_STEP, APPROVAL_STEP} & set(setting("STEPS"))):
        errors.append(
            Error(
                f"AI_SDK_WORKFLOWS['STEPS'] has the key {reserved!r}, which is reserved for "
                "a built-in step type, so a definition can never reach that class.",
                hint="Register the step under another key.",
                id=f"{PREFIX}.E004",
            )
        )
    for name, base, number in (("STEPS", Step, "E002"), ("LISTENERS", RunListener, "E003")):
        _, problems = load_classes(name, base)
        errors.extend(
            Error(
                problem,
                hint="Fix the import path, or remove the key. A definition naming it "
                "cannot be stored or run.",
                id=f"{PREFIX}.{number}",
            )
            for problem in problems
        )
    return errors


# What each AI_SDK_WORKFLOWS key accepts: a test, and the words E012 shows when it fails.
ACCEPTS: dict[str, tuple[Callable[[Any], bool], str]] = {
    "STEPS": (lambda value: isinstance(value, dict), "a dict of step classes by key"),
    "LISTENERS": (lambda value: isinstance(value, dict), "a dict of listener classes by key"),
    "SCHEDULES_ENABLED": (lambda value: isinstance(value, bool), "True or False"),
    "STALE_AFTER_SECONDS": (
        lambda value: isinstance(value, int) and not isinstance(value, bool) and value > 0,
        "a positive number of seconds",
    ),
}


def settings_errors() -> list[Error]:
    """E012: AI_SDK_WORKFLOWS is a dict of known keys, each of the kind it takes."""
    from django_ai_sdk.checks import dict_setting_errors

    return dict_setting_errors("AI_SDK_WORKFLOWS", ACCEPTS, PREFIX)


__all__ = ["PREFIX", "check_workflows"]
