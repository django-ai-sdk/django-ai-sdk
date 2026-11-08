"""The AI_SDK_WORKFLOWS setting, read at call time. docs/content/manual/settings.md lists it."""

from __future__ import annotations

import logging

from django.utils.module_loading import import_string

from django_ai_sdk.checks import AppSettings, of_type

logger = logging.getLogger(__name__)

setting = AppSettings(
    "AI_SDK_WORKFLOWS",
    {
        "STEPS": (
            {},
            of_type(dict, label="a dict"),
            'Step classes by key: {"ocr": "myapp.steps.OcrStep"}.',
        ),
        "LISTENERS": (
            {},
            of_type(dict, label="a dict"),
            'Listener classes by key: {"log": "myapp.listeners.LogListener"}.',
        ),
    },
)


def load_classes[T](name: str, base: type[T]) -> tuple[dict[str, type[T]], list[str]]:
    """Import the classes listed under AI_SDK_WORKFLOWS[name].

    Returns the classes by key, plus a problem message for each path that does not
    import or is not a `base` subclass. The system check reports the problems. At
    run time those keys are just missing.
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


__all__ = ["load_classes", "registered", "setting"]
