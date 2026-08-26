"""Per-automation configuration from AI_SDK_AUTOMATIONS, read on every tick and never raising.

`enabled` resolves DB row > settings > class attribute; `cron` and `timezone` have no DB layer.
"""

from __future__ import annotations

import logging
from typing import Any

from django_ai_sdk.utils import resolve_setting

logger = logging.getLogger(__name__)


def _configured() -> dict[str, Any]:
    """AI_SDK_AUTOMATIONS, or an empty dict if it is not one."""
    configured = resolve_setting("AI_SDK_AUTOMATIONS") or {}
    if isinstance(configured, dict):
        return configured
    logger.error(
        "AI_SDK_AUTOMATIONS must be a dict of {name: {KEY: value}}, got %s; every "
        "automation keeps its own defaults",
        type(configured).__name__,
    )
    return {}


def get_automation_config(name: str) -> dict[str, Any]:
    """This automation's slice of AI_SDK_AUTOMATIONS, with upper-cased keys."""
    entry = _configured().get(name) or {}
    if not isinstance(entry, dict):
        logger.error(
            "AI_SDK_AUTOMATIONS[%r] must be a dict of {KEY: value}, got %s; %r keeps "
            "its own defaults",
            name,
            type(entry).__name__,
            name,
        )
        return {}
    return {str(key).upper(): value for key, value in entry.items()}


def configured_names() -> set[str]:
    """Every automation name AI_SDK_AUTOMATIONS has an entry for."""
    return set(_configured())


def automations_enabled() -> bool:
    """The global kill switch, AI_SDK_AUTOMATIONS_ENABLED."""
    return bool(resolve_setting("AI_SDK_AUTOMATIONS_ENABLED", True))


def lease_seconds() -> int:
    """How long a claimed automation stays locked, AI_SDK_AUTOMATION_LEASE.

    The bound on how long a crashed worker blocks the next run.
    """
    return int(resolve_setting("AI_SDK_AUTOMATION_LEASE", 3600))


def is_enabled(name: str, *, code_default: bool, db_value: bool | None) -> tuple[bool, str]:
    """Resolve enabled-ness and report which layer decided it."""
    if not automations_enabled():
        return False, "kill-switch"
    if db_value is not None:
        return db_value, "db"
    configured = get_automation_config(name).get("ENABLED")
    if configured is not None:
        return bool(configured), "settings"
    return code_default, "code"


__all__ = [
    "automations_enabled",
    "configured_names",
    "get_automation_config",
    "is_enabled",
    "lease_seconds",
]
