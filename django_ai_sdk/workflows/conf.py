"""The AI_SDK_WORKFLOWS setting, read at call time. docs/content/manual/settings.md lists it."""

from __future__ import annotations

import logging
from typing import Any

from django.utils.module_loading import import_string

from django_ai_sdk.oversight.conf import setting_from

logger = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    # Step classes by key: {"ocr": "myapp.steps.OcrStep"}.
    "STEPS": {},
    # Listener classes by key: {"log": "myapp.listeners.LogListener"}.
    "LISTENERS": {},
    # False: a cron occurrence starts nothing, such as on a staging copy of production.
    "SCHEDULES_ENABLED": True,
    # Idle running runs older than this are failed if DBOS no longer runs them.
    "STALE_AFTER_SECONDS": 900,
}


def setting(key: str) -> Any:
    return setting_from("AI_SDK_WORKFLOWS", DEFAULTS, key)


def load_classes[T](name: str, base: type[T]) -> tuple[dict[str, type[T]], list[str]]:
    """Import AI_SDK_WORKFLOWS[name] classes; return them by key plus a problem per bad path.

    Bad keys are just missing at run time; the system check reports them.
    """
    classes: dict[str, type[T]] = {}
    problems: list[str] = []
    for key, path in (setting(name) or {}).items():
        where = f"AI_SDK_WORKFLOWS[{name!r}][{key!r}] names {path!r}"
        try:
            cls = import_string(path)
        except ImportError as exc:
            problems.append(f"{where}, which does not import: {exc}")
            continue
        if not isinstance(cls, type) or not issubclass(cls, base):
            problems.append(f"{where}, which is not a {base.__name__} subclass.")
            continue
        classes[key] = cls
    return classes, problems


def registered[T](name: str, base: type[T]) -> dict[str, type[T]]:
    """`load_classes`, logging each problem. The system check reports them too."""
    classes, problems = load_classes(name, base)
    for problem in problems:
        logger.warning("%s It is not composable.", problem)
    return classes


__all__ = ["DEFAULTS", "load_classes", "registered", "setting"]
