"""The AI_SDK_OVERSIGHT setting. Only this module reads it; check E012 validates it."""

from __future__ import annotations

from typing import Any

from django.apps import apps
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string

from django_ai_sdk.oversight.types import Capture

APP_LABEL = "django_ai_sdk_oversight"

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
    # A tool result or source longer than a judge reads is judged in chunks this long,
    # at most this many; beyond that it is too long, as before.
    "OTHER_TEXT_CHUNK_CHARS": 5_000,
    "MAX_OTHER_TEXT_CHUNKS": 10,
    "CLIENT_DETAIL": "actions",
    "FAIL_MODE": "closed",
    "RETENTION": {"content": 90, "metadata": 730},
    # Notifiers as JUDGES names its judges: ["LogNotifier"].
    "NOTIFIERS": [],
    # How long each process caches authored policies. 0 reads them every turn.
    "RUNTIME_CACHE_SECONDS": 10,
}


def configured(name: str = "AI_SDK_OVERSIGHT") -> Any:
    """{} when unset. It may not be a dict."""
    return getattr(settings, name, None) or {}


def setting_from(name: str, defaults: dict[str, Any], key: str) -> Any:
    """A dict value merges one level deep into a dict default."""
    user_settings = configured(name)
    if not isinstance(user_settings, dict):
        raise ImproperlyConfigured(f"{name} must be a dict, not a {type(user_settings).__name__}.")
    default = defaults[key]
    value = user_settings.get(key, default)
    if isinstance(value, dict) and isinstance(default, dict):
        return {**default, **value}
    return value


def setting(key: str) -> Any:
    return setting_from("AI_SDK_OVERSIGHT", DEFAULTS, key)


def build_configured[T](spec: Any, base: type[T], package: str) -> T:
    """A `base` from an instance, a class name, or {"class": class, **kwargs}.

    A class name is a dotted path, or a bare name of one of `package`'s own classes.
    """
    if isinstance(spec, base):
        return spec
    config = {"class": spec} if isinstance(spec, str) else spec
    if not isinstance(config, dict):
        raise ImproperlyConfigured(f"{spec!r} is not a {base.__name__}, a class or a config")
    config = dict(config)
    path = config.pop("class", "")
    if not path:
        raise ImproperlyConfigured(f"{base.__name__} config {config!r} has no 'class'")
    if isinstance(path, str) and "." not in path:
        path = f"{package}.{path}"
    cls = import_string(path) if isinstance(path, str) else path
    if not (isinstance(cls, type) and issubclass(cls, base)):
        raise ImproperlyConfigured(f"{path!r} is not a {base.__name__} subclass")
    return cls(**config)


def is_retention(value: Any) -> bool:
    return isinstance(value, dict) and all(
        kind in RETENTION_KINDS and type(days) is int and days >= 1 for kind, days in value.items()
    )


def validate_retention(value: Any) -> dict[str, int]:
    """Raise ValueError; the system check does not stop a turn or prune, so runtime checks too."""
    if not is_retention(value):
        raise ValueError(
            f"retention must map 'content' and/or 'metadata' to whole days of at least 1, not {value!r}"
        )
    return dict(value)


def deadline_ms(stage: str) -> float | None:
    return (setting("DEADLINE_MS") or {}).get(str(stage))


def is_installed() -> bool:
    return apps.is_installed("django_ai_sdk.oversight")


def is_enabled() -> bool:
    return is_installed() and bool(setting("ENABLED"))
