"""System checks for workflows. docs/content/manual/workflows.md lists the ids."""

from __future__ import annotations

from typing import Any

from django.core.checks import Error, Warning

PREFIX = "ai_sdk.workflows"


def check_workflows(app_configs: Any = None, **kwargs: Any) -> list[Any]:
    """Report every declaration and every registry entry that cannot be used."""
    from django.conf import settings as django_settings

    from django_ai_sdk.workflows.conf import load_classes, setting
    from django_ai_sdk.workflows.listeners import RunListener
    from django_ai_sdk.workflows.registry import get_invalid_workflows
    from django_ai_sdk.workflows.schemas import AGENT_STEP
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
    settings = setting.errors(PREFIX)
    if settings:
        # The registries below read the same setting.
        return [*errors, *settings]
    if AGENT_STEP in setting("STEPS"):
        errors.append(
            Error(
                f"AI_SDK_WORKFLOWS['STEPS'] has the key {AGENT_STEP!r}, which is reserved for "
                "agent steps, so a definition can never reach that class.",
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


def check_workflows_deploy(app_configs: Any = None, **kwargs: Any) -> list[Any]:
    """Warn when the task backend runs workflow runs inline, or never runs them."""
    from django_ai_sdk.checks import TASKS_HINT, task_backend_problem
    from django_ai_sdk.workflows.tasks import execute_workflow

    if problem := task_backend_problem(execute_workflow):
        return [Warning(f"Workflow runs {problem}.", hint=TASKS_HINT, id=f"{PREFIX}.W001")]
    return []


__all__ = ["PREFIX", "check_workflows", "check_workflows_deploy"]
