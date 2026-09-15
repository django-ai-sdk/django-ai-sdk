from __future__ import annotations

from typing import Any

from django.conf import settings


def resolve_setting(setting_name: str, default: Any = None) -> Any:
    """Read a Django setting, falling back to `default` when it is unset."""
    return getattr(settings, setting_name, default)


def serialize(value: Any) -> Any:
    """JSON-safe form of a value, recursing into lists and dicts of pydantic models."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple):
        return [serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: serialize(item) for key, item in value.items()}
    dump = getattr(value, "model_dump", None)
    return dump(mode="json") if callable(dump) else str(value)
