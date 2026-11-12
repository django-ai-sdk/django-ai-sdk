"""The oversight app. On start it loads the packs, each app's `oversight.py`, the
system checks and the deletion signals.

The opt-in apps on top of it call `require_apps` and `extend` from their own `ready()`,
so core never names them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django.apps import AppConfig
from django.core.checks import Tags
from django.core.checks import register as register_check
from django.utils.module_loading import autodiscover_modules

if TYPE_CHECKING:
    from collections.abc import Mapping


class OversightConfig(AppConfig):
    """Opt-in app that checks and records agent turns."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "django_ai_sdk.oversight"
    label = "django_ai_sdk_oversight"
    verbose_name = "Oversight"

    def ready(self) -> None:
        # Load the packs first so an app's policy can extend them by name.
        import django_ai_sdk.oversight.packs  # noqa: F401

        autodiscover_modules("oversight")

        from django_ai_sdk.oversight import signals
        from django_ai_sdk.oversight.checks import (
            check_oversight,
            check_oversight_deploy,
            check_oversight_retention,
        )

        register_check(check_oversight)
        register_check(check_oversight_deploy, deploy=True)
        register_check(check_oversight_retention, Tags.database, deploy=True)
        # Deleting a thread or a user forgets what oversight holds about it.
        signals.connect()


def require_apps(app_name: str, *needed: str) -> None:
    """Refuse to start `app_name` without each app in `needed`."""
    from django.apps import apps
    from django.core.exceptions import ImproperlyConfigured

    for name in needed:
        if not apps.is_installed(name):
            raise ImproperlyConfigured(f"{app_name} needs {name} in INSTALLED_APPS")


# What the installed opt-in apps added (`extend`).
MODEL_PERMISSIONS: dict[Any, str] = {}
ROUTE_MODULES: list[str] = []


def extend(*, permissions: Mapping[Any, str] | None = None, routes: str = "") -> None:
    """Add an opt-in app's operations and routes. Call it from the app's `ready()`.

    `permissions` maps each operation the app brings to the model permission
    `OversightDefaultPermission` checks for it. `routes` is the dotted path of a module
    whose `routes` table `get_oversight_router()` mounts. Calling it again is harmless.
    """
    MODEL_PERMISSIONS.update(permissions or {})
    if routes and routes not in ROUTE_MODULES:
        ROUTE_MODULES.append(routes)
