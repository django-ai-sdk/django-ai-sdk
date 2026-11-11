"""System checks for authoring. docs/content/oversight-authoring.md lists the ids."""

from __future__ import annotations

from typing import Any

from django.core.checks import Error, Warning

PREFIX = "ai_sdk.oversight.authoring"


def check_authoring(app_configs: Any = None, **kwargs: Any) -> list[Any]:
    """E012, then E001."""
    from django_ai_sdk.oversight.authoring.conf import setting
    from django_ai_sdk.oversight.authoring.definitions import is_authored
    from django_ai_sdk.oversight.registry import get_policy_class

    found = setting.errors(PREFIX)
    if found:
        return found
    for name in setting("MUST_EXTEND"):
        cls = get_policy_class(name)
        if cls is None or is_authored(cls):
            found.append(
                Error(
                    f'AI_SDK_OVERSIGHT_AUTHORING["MUST_EXTEND"] names {name!r}, which no code registers.',
                    hint="Name a pack or a policy declared in code, such as 'security'.",
                    id=f"{PREFIX}.E001",
                )
            )
    return found


def check_authoring_deploy(app_configs: Any = None, **kwargs: Any) -> list[Any]:
    """W001. Registered with deploy=True: Django passes no deploy flag to the check."""
    from django_ai_sdk.oversight.authoring.conf import setting

    if setting.broken or setting("MUST_EXTEND"):
        return []
    return [
        Warning(
            "Authored policies need not extend any policy: an operator can define one "
            "without the protective rules code declares, for every agent that names it.",
            hint='Set AI_SDK_OVERSIGHT_AUTHORING["MUST_EXTEND"], such as ["security"].',
            id=f"{PREFIX}.W001",
        )
    ]


__all__ = ["PREFIX", "check_authoring", "check_authoring_deploy"]
