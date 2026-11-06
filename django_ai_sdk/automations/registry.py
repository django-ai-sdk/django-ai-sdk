"""Process-wide registry of declared automations.

Each installed app's `automations` module is imported on startup, so decorating a
subclass with `register_automation` there is enough. An invalid declaration is left
out and reported by the system check, so one app's typo does not stop the site from
booting.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from django.core.exceptions import ImproperlyConfigured

if TYPE_CHECKING:
    from django_ai_sdk.automations.base import Automation

logger = logging.getLogger(__name__)

_registry: dict[str, Automation] = {}

# Rejected declarations, name -> reason. Read by the system check.
_invalid: dict[str, str] = {}


def register_automation[T: type[Automation]](cls: T) -> T:
    """Register an Automation class unless its declaration is invalid.

    Returns the class unchanged either way, so it stays importable.
    """
    automation = cls()
    try:
        validate(automation)
    except ImproperlyConfigured as exc:
        _invalid[automation.name or cls.__name__] = str(exc)
        logger.warning("Automation not registered: %s", exc)
        return cls

    _invalid.pop(automation.name, None)
    if (existing := _registry.get(automation.name)) and type(existing) is not cls:
        # The later one wins, and which is later depends on app-loading order.
        logger.warning(
            "Automation %r is declared by both %s and %s. Give one a different `name`.",
            automation.name,
            type(existing).__name__,
            cls.__name__,
        )
    _registry[automation.name] = automation
    return cls


def validate(automation: Automation) -> None:
    """Raise ImproperlyConfigured unless the declaration is complete."""
    if not automation.name:
        raise ImproperlyConfigured(
            f"{type(automation).__name__} must set a non-empty `name` to be registered."
        )

    if not automation.workflow:
        raise ImproperlyConfigured(
            f"Automation {automation.name!r} names no workflow. Set `workflow` to the "
            "name of a workflow registered in an app's workflows.py, or a stored "
            "Workflow slug."
        )

    # An unknown workflow is reported by check W003 instead. Registration must not
    # touch the database, and a workflow may live only there.

    # Validates the cron expression and its timezone.
    automation.get_schedule()


def get_invalid_automations() -> dict[str, str]:
    """Declarations rejected by validate(), name -> reason. Read by the system check."""
    return dict(_invalid)


def get_automations() -> dict[str, Automation]:
    """Every registered automation, keyed by name."""
    return dict(_registry)


def get_automation(name: str) -> Automation | None:
    """One automation by name, or None."""
    return _registry.get(name)


def reset_registry() -> None:
    """Clear the registry — for tests."""
    _registry.clear()
    _invalid.clear()


__all__ = [
    "get_automation",
    "get_automations",
    "get_invalid_automations",
    "register_automation",
    "reset_registry",
    "validate",
]
