"""The authoring app: its AI_SDK_OVERSIGHT_AUTHORING setting, its system checks, and what it
adds to oversight on start (permissions, routes and the policy loader).

docs/content/oversight-authoring.md lists the setting's keys and the check ids.
"""

from __future__ import annotations

from typing import Any

from django.apps import AppConfig
from django.core.checks import Error, Warning
from django.core.checks import register as register_check
from pydantic import BaseModel, ConfigDict, StrictStr

from django_ai_sdk.oversight.conf import AppSettings

PREFIX = "ai_sdk.oversight.authoring"


class AuthoringSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Policies every authored policy must extend, such as ["security"].
    MUST_EXTEND: list[StrictStr] = []


setting = AppSettings("AI_SDK_OVERSIGHT_AUTHORING", AuthoringSettings)


class AuthoringConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "django_ai_sdk.oversight.authoring"
    label = "django_ai_sdk_oversight_authoring"
    verbose_name = "Oversight authoring"

    def ready(self) -> None:
        from django_ai_sdk.oversight.apps import extend, require_apps
        from django_ai_sdk.permissions import Operation

        require_apps(self.name, "django_ai_sdk.oversight")
        extend(
            permissions={
                Operation.VIEW_OVERSIGHT_AUTHORING: f"{self.label}.view_authoredpolicy",
                Operation.AUTHOR_OVERSIGHT: f"{self.label}.change_authoredpolicy",
            },
            routes="django_ai_sdk.contrib.ninja.oversight_authoring",
        )
        from django_ai_sdk.oversight.authoring.loader import loader
        from django_ai_sdk.oversight.registry import add_policy_loader

        register_check(check_authoring)
        register_check(check_authoring_deploy, deploy=True)
        add_policy_loader(loader.load)


def check_authoring(app_configs: Any = None, **kwargs: Any) -> list[Any]:
    """E012, then E001."""
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
