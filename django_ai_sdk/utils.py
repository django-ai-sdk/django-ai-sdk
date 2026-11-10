from __future__ import annotations

import contextlib
import uuid
from typing import TYPE_CHECKING, Any

from django.conf import settings

if TYPE_CHECKING:
    import contextvars
    from collections.abc import Callable, Generator


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


def isoformat_or_none(value: Any) -> str | None:
    """ISO 8601 text of a date or datetime, or None when there is none."""
    return value.isoformat() if value else None


def as_uuid(value: Any) -> uuid.UUID | None:
    """`value` as a UUID, or None when it is empty or not one."""
    if not value:
        return None
    try:
        return uuid.UUID(str(value))
    except ValueError:
        return None


@contextlib.contextmanager
def scoped[T](var: contextvars.ContextVar[T], value: T) -> Generator[None]:
    """Set `var` to `value` for the block, then put the old value back."""
    token = var.set(value)
    try:
        yield
    finally:
        var.reset(token)


def map_strings(value: Any, change: Callable[[str], str], *, keys: bool = False) -> Any:
    """Apply `change` to every string in a JSON value, and to its keys when `keys` is set."""
    if isinstance(value, str):
        return change(value)
    if isinstance(value, dict):
        return {
            (change(k) if keys and isinstance(k, str) else k): map_strings(v, change, keys=keys)
            for k, v in value.items()
        }
    if isinstance(value, list | tuple):
        return [map_strings(item, change, keys=keys) for item in value]
    return value
