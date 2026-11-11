"""The AI_SDK_OVERSIGHT setting. Only this module reads it; docs/content/manual/oversight.md lists the keys."""

from __future__ import annotations

from typing import Any

from django.apps import apps

from django_ai_sdk.checks import AppSettings, number, of_type, one_of
from django_ai_sdk.oversight.types import Capture

APP_LABEL = "django_ai_sdk_oversight"

STAGE_KEYS = ("gate", "tool", "finish")
# Allowed values, shared with policy.py, which checks a policy's own options.
STREAM_MODES = ("auto", "live")
FAIL_MODES = ("closed", "open")
RETENTION_KEYS = ("content", "metadata")


def _text(value: Any) -> str:
    return "" if isinstance(value, str) else f"is {value!r}, not a policy name"


_positive = number(0, above=True, allow_none=True)


def _judges(value: Any) -> str:
    if not isinstance(value, dict):
        return f'is a {type(value).__name__}, not a dict of aliases: {{"judge": {{"class": ...}}}}'
    from django_ai_sdk.oversight.judges.base import Judge

    bad = [alias for alias, spec in value.items() if not isinstance(spec, str | dict | Judge)]
    return f"has aliases that are not judge specs: {bad}" if bad else ""


def _deadlines(value: Any) -> str:
    if not isinstance(value, dict):
        return f'is {value!r}, not a dict of milliseconds per stage: {{"gate": 1000}}'
    unknown = sorted(str(stage) for stage in value if stage not in STAGE_KEYS)
    if unknown:
        return f"names unknown stages {unknown}; the stages are {list(STAGE_KEYS)}"
    bad = {stage: ms for stage, ms in value.items() if _positive(ms)}
    return f"has values that are not positive milliseconds or None: {bad}" if bad else ""


def _retention(value: Any) -> str:
    """The problem with a retention dict, or "". `validate_retention` raises it at runtime."""
    if not isinstance(value, dict):
        return f'is {value!r}, not {{"content": days, "metadata": days}}'
    unknown = sorted(str(kind) for kind in value if kind not in RETENTION_KEYS)
    if unknown:
        return f'names unknown kinds {unknown}; the kinds are ["content", "metadata"]'
    bad = {
        kind: days
        for kind, days in value.items()
        if isinstance(days, bool) or not isinstance(days, int) or days < 1
    }
    return f"has values that are not whole days of at least 1: {bad}" if bad else ""


def validate_retention(value: Any) -> dict[str, int]:
    """Return `value` as retention days, or raise ValueError with the system check's message.

    The system check does not stop a turn or a prune, so runtime checks the same rule again.
    """
    problem = _retention(value)
    if problem:
        raise ValueError(f"retention {problem}")
    return dict(value)


setting = AppSettings(
    "AI_SDK_OVERSIGHT",
    {
        # Kill switch: False leaves every agent unguarded.
        "ENABLED": (
            True,
            of_type(bool, label="True or False"),
            "True runs oversight, False turns it off everywhere.",
        ),
        # Policy for agents without `oversight`; "" leaves them unguarded.
        "DEFAULT_POLICY": ("", _text, 'A registered policy\'s name, or "" for none.'),
        # The most content the record keeps. A policy may keep less, never more.
        "CAPTURE": (
            Capture.MASKED,
            one_of(*(c.value for c in Capture)),
            'Use "masked" (the default).',
        ),
        "STREAM_MODE": ("auto", one_of(*STREAM_MODES), 'Use "auto" (the default) or "live".'),
        "JUDGES": ({}, _judges, 'Name judges by alias: {"judge": {"class": "LLMJudge"}}.'),
        # Milliseconds per inline stage; a lens still running at the deadline fails.
        "DEADLINE_MS": (
            {"gate": 1000, "tool": 1000, "finish": 2_000},
            _deadlines,
            'Milliseconds per stage: {"gate": 1000, "tool": 1000, "finish": 2000}.',
        ),
        # Longer questions are refused at the gate: lenses would only read part of them.
        "MAX_INPUT_CHARS": (
            100_000,
            _positive,
            "A number of characters, or None for no limit.",
        ),
        "CLIENT_DETAIL": (
            "actions",
            one_of("actions", "full"),
            'Use "actions" (the default) or "full".',
        ),
        "FAIL_MODE": (
            "closed",
            one_of(*FAIL_MODES),
            'Use "closed" (the default) or "open"; a policy may set its own fail_mode.',
        ),
        "RETENTION": (
            {"content": 90, "metadata": 730},
            _retention,
            '{"content": 90, "metadata": 730}, either key alone.',
        ),
        "NOTIFIERS": (
            [],
            of_type(list, tuple, label="a list of dotted paths or Notifier instances"),
            'A list: ["myapp.notifiers.Slack", WebhookNotifier(url)].',
        ),
        # How long each process caches authored policies. 0 reads them every turn.
        "RUNTIME_CACHE_SECONDS": (10, number(0), "Seconds, such as 10."),
    },
)


def deadline_ms(stage: str) -> float | None:
    """The deadline in milliseconds for `stage`, or None for no limit."""
    return (setting("DEADLINE_MS") or {}).get(str(stage))


def is_installed() -> bool:
    return apps.is_installed("django_ai_sdk.oversight")


def is_enabled() -> bool:
    """Whether the app is installed and ENABLED is true."""
    return is_installed() and bool(setting("ENABLED"))


__all__ = [
    "APP_LABEL",
    "deadline_ms",
    "is_enabled",
    "is_installed",
    "setting",
    "validate_retention",
]
