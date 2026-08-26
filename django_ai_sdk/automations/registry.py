"""Process-wide registry of declared automations.

Every installed app's `automations` module is imported on startup, so decorating a
subclass with `register` there is enough. A declaration that cannot run is kept out of
the registry and reported by the `django_ai_sdk.automations` check, so one app's typo
does not stop the site from booting.
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

# Warned once per process; this sits on the read path.
_checked_orphans = False


def register[T: type[Automation]](cls: T) -> T:
    """Class decorator: add an Automation to the registry unless it cannot run.

    Returns the class unchanged either way, so a declaration stays importable.
    """
    automation = cls()
    try:
        validate(automation)
    except ImproperlyConfigured as exc:
        _invalid[automation.name or cls.__name__] = str(exc)
        logger.warning("Automation not registered: %s", exc)
        return cls

    _invalid.pop(automation.name, None)
    existing = _registry.get(automation.name)
    if existing is not None and type(existing) is not cls:
        logger.warning(
            "Automation %r is declared by both %s and %s — only one is reachable under "
            "that name, and which one depends on app-loading order. Give one of them a "
            "different `name`.",
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
            "name of one declared in an app's workflows.py."
        )

    # Whether that name resolves is W003, not an error here: registration must not
    # touch the database, and a workflow may live only there.

    # Validates the cron expression.
    automation.get_schedule()


def get_invalid_automations() -> dict[str, str]:
    """Declarations rejected by validate(), name -> reason. Read by the system check."""
    return dict(_invalid)


def get_automations() -> dict[str, Automation]:
    """Every registered automation, keyed by name."""
    _warn_orphaned_config()
    return dict(_registry)


def get_automation(name: str) -> Automation | None:
    """One automation by name, or None."""
    _warn_orphaned_config()
    return _registry.get(name)


def _warn_orphaned_config() -> None:
    """Name AI_SDK_AUTOMATIONS entries that nothing declares."""
    global _checked_orphans
    if _checked_orphans:
        return
    _checked_orphans = True

    from django_ai_sdk.automations.config import configured_names

    orphaned = configured_names() - set(_registry)
    if orphaned:
        logger.warning(
            "AI_SDK_AUTOMATIONS configures %s, but nothing declares them; check the "
            "spelling, or that the declaring app is in INSTALLED_APPS",
            ", ".join(sorted(repr(name) for name in orphaned)),
        )


def reset_registry() -> None:
    """Clear the registry — for tests."""
    global _checked_orphans
    _registry.clear()
    _invalid.clear()
    _checked_orphans = False


__all__ = [
    "get_automation",
    "get_automations",
    "get_invalid_automations",
    "register",
    "reset_registry",
    "validate",
]
