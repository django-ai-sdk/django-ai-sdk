"""Startup diagnostics for declared workflows.

Reported as an Error: an unrunnable workflow is a defect, and `manage.py check` should
fail on it in a deployment pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django.core.checks import Error

if TYPE_CHECKING:
    from collections.abc import Sequence

ID_PREFIX = "ai_sdk.workflows"


def check_workflows(app_configs: Any = None, **kwargs: Any) -> Sequence[Any]:
    """Report every declaration register() refused, one Error each."""
    from django_ai_sdk.workflows.registry import get_invalid_workflows

    return [
        Error(
            reason,
            hint="Fix the declaration in the app's workflows module.",
            id=f"{ID_PREFIX}.E001",
        )
        for _, reason in sorted(get_invalid_workflows().items())
    ]


__all__ = ["check_workflows"]
