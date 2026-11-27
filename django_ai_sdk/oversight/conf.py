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
    "ENABLED": True,
    "DEFAULT_POLICY": "",
    "CAPTURE": Capture.MASKED,
    "STREAM_MODE": "auto",
    "JUDGES": {},
    "DEADLINE_MS": {"gate": 1000, "tool": 1000, "finish": 2_000},
    "MAX_INPUT_CHARS": 100_000,
    "MAX_OTHER_TEXT_CHUNKS": 10,
    "CLIENT_DETAIL": "actions",
    "FAIL_MODE": "closed",
    "RETENTION": {"content": 90, "metadata": 730},
    "NOTIFIERS": [],
    "RUNTIME_CACHE_SECONDS": 10,
    "UNGUARDED_MODEL_CALLS": "refuse",
}

UNGUARDED_MODES = ("refuse", "warn", "allow")


def configured(name: str = "AI_SDK_OVERSIGHT") -> Any:
    return getattr(settings, name, None) or {}


def setting_from(name: str, defaults: dict[str, Any], key: str) -> Any:
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
