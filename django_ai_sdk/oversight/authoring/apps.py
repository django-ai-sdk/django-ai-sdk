"""The authoring app: its setting, its system checks, and what it adds to oversight on start."""

from __future__ import annotations

from typing import Any

from django.apps import AppConfig
from django.core.checks import Error, Warning
from django.core.checks import register as register_check

from django_ai_sdk.oversight.conf import configured, setting_from

PREFIX = "ai_sdk.oversight.authoring"
SETTING = "AI_SDK_OVERSIGHT_AUTHORING"

AUTHORING_DEFAULTS: dict[str, Any] = {
    # Policies every authored policy must extend, such as ["security"].
    "MUST_EXTEND": [],
}


def authoring_setting(key: str) -> Any:
    return setting_from(SETTING, AUTHORING_DEFAULTS, key)


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

    found = settings_checks()
    if found:
        return found
    for name in authoring_setting("MUST_EXTEND"):
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


def settings_checks() -> list[Error]:
    user_settings = configured(SETTING)
    if not isinstance(user_settings, dict):
        return [
            Error(
                f"AI_SDK_OVERSIGHT_AUTHORING must be a dict, not a {type(user_settings).__name__}.",
                hint="Set a dict, or remove it.",
                id=f"{PREFIX}.E012",
            )
        ]
    errors = [
        Error(
            f"AI_SDK_OVERSIGHT_AUTHORING has an unknown key {key!r}.",
            hint=f"The keys are {', '.join(AUTHORING_DEFAULTS)}.",
            id=f"{PREFIX}.E012",
        )
        for key in user_settings
        if key not in AUTHORING_DEFAULTS
    ]
    must_extend = user_settings.get("MUST_EXTEND", [])
    if not isinstance(must_extend, list | tuple) or not all(
        isinstance(name, str) for name in must_extend
    ):
        errors.append(
            Error(
                "AI_SDK_OVERSIGHT_AUTHORING['MUST_EXTEND'] must be a list of policy names, "
                "such as ['security'].",
                id=f"{PREFIX}.E012",
            )
        )
    return errors


def check_authoring_deploy(app_configs: Any = None, **kwargs: Any) -> list[Any]:
    """W001. Registered with deploy=True: Django passes no deploy flag to the check."""
    if not isinstance(configured(SETTING), dict) or authoring_setting("MUST_EXTEND"):
        return []
    return [
        Warning(
            "Authored policies need not extend any policy: an operator can define one "
            "without the protective rules code declares, for every agent that names it.",
            hint='Set AI_SDK_OVERSIGHT_AUTHORING["MUST_EXTEND"], such as ["security"].',
            id=f"{PREFIX}.W001",
        )
    ]
