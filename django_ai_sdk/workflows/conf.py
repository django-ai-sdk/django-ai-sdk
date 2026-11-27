from __future__ import annotations

import logging
from typing import Any

from django.utils.module_loading import import_string

from django_ai_sdk.oversight.conf import setting_from

logger = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "STEPS": {},
    "LISTENERS": {},
    "SCHEDULES_ENABLED": True,
    "STALE_AFTER_SECONDS": 900,
}


def setting(key: str) -> Any:
    return setting_from("AI_SDK_WORKFLOWS", DEFAULTS, key)


def load_classes[T](name: str, base: type[T]) -> tuple[dict[str, type[T]], list[str]]:
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
    classes, problems = load_classes(name, base)
    for problem in problems:
        logger.warning("%s It is not composable.", problem)
    return classes


__all__ = ["DEFAULTS", "load_classes", "registered", "setting"]
