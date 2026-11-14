"""The AI_SDK_OVERSIGHT setting. Only this module reads it; docs/content/manual/oversight.md lists the keys.

The E012 check in checks.py validates it.
"""

from __future__ import annotations

from typing import Any

from django.apps import apps
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.oversight.types import Capture

APP_LABEL = "django_ai_sdk_oversight"

# Allowed values, shared with policy.py, which checks a policy's own options.
STREAM_MODES = ("auto", "live")
FAIL_MODES = ("closed", "open")
RETENTION_KINDS = ("content", "metadata")

DEFAULTS: dict[str, Any] = {
    # Kill switch: False leaves every agent unguarded.
    "ENABLED": True,
    # Policy for agents without `oversight`; "" leaves them unguarded.
    "DEFAULT_POLICY": "",
    # The most content the record keeps. A policy may keep less, never more.
    "CAPTURE": Capture.MASKED,
    "STREAM_MODE": "auto",
    # Judges by alias: {"judge": {"class": "LLMJudge"}}.
    "JUDGES": {},
    # Milliseconds per inline stage; a lens still running at the deadline fails.
    "DEADLINE_MS": {"gate": 1000, "tool": 1000, "finish": 2_000},
    # Longer questions are blocked at the gate: lenses would only read part of them.
    "MAX_INPUT_CHARS": 100_000,
    "CLIENT_DETAIL": "actions",
    "FAIL_MODE": "closed",
    "RETENTION": {"content": 90, "metadata": 730},
    # Dotted paths or Notifier instances.
    "NOTIFIERS": [],
    # How long each process caches authored policies. 0 reads them every turn.
    "RUNTIME_CACHE_SECONDS": 10,
}


def configured(name: str = "AI_SDK_OVERSIGHT") -> Any:
    """The setting `name` as the project set it, {} when unset. It may not be a dict."""
    return getattr(settings, name, None) or {}


def setting_from(name: str, defaults: dict[str, Any], key: str) -> Any:
    """The setting `name`'s `key`, or its default in `defaults`.

    A dict value merges one level deep into a dict default.
    """
    user_settings = configured(name)
    if not isinstance(user_settings, dict):
        raise ImproperlyConfigured(f"{name} must be a dict, not a {type(user_settings).__name__}.")
    default = defaults[key]
    value = user_settings.get(key, default)
    if isinstance(value, dict) and isinstance(default, dict):
        return {**default, **value}
    return value


def setting(key: str) -> Any:
    """AI_SDK_OVERSIGHT[key], or its default. A dict value merges one level deep into a dict default."""
    return setting_from("AI_SDK_OVERSIGHT", DEFAULTS, key)


def is_retention(value: Any) -> bool:
    """Whether `value` maps "content" and/or "metadata" to whole days of at least 1."""
    return isinstance(value, dict) and all(
        kind in RETENTION_KINDS and type(days) is int and days >= 1 for kind, days in value.items()
    )


def validate_retention(value: Any) -> dict[str, int]:
    """Return `value` as retention days, or raise ValueError.

    The system check does not stop a turn or a prune, so runtime checks the same rule again.
    """
    if not is_retention(value):
        raise ValueError(
            f"retention must map 'content' and/or 'metadata' to whole days of at least 1, not {value!r}"
        )
    return dict(value)


def deadline_ms(stage: str) -> float | None:
    """The deadline in milliseconds for `stage`, or None for no limit."""
    return (setting("DEADLINE_MS") or {}).get(str(stage))


def is_installed() -> bool:
    return apps.is_installed("django_ai_sdk.oversight")


def is_enabled() -> bool:
    """Whether the app is installed and ENABLED is true."""
    return is_installed() and bool(setting("ENABLED"))
