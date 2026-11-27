from __future__ import annotations

import contextlib
import uuid
from typing import TYPE_CHECKING, Any

from django.conf import settings

if TYPE_CHECKING:
    import contextvars
    from collections.abc import Callable, Generator

    from django.contrib.auth.base_user import AbstractBaseUser


def resolve_setting(setting_name: str, default: Any = None) -> Any:
    return getattr(settings, setting_name, default)


def serialize(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple):
        return [serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: serialize(item) for key, item in value.items()}
    dump = getattr(value, "model_dump", None)
    return dump(mode="json") if callable(dump) else str(value)


def isoformat_or_none(value: Any) -> str | None:
    return value.isoformat() if value else None


def as_uuid(value: Any) -> uuid.UUID | None:
    if not value:
        return None
    try:
        return uuid.UUID(str(value))
    except ValueError:
        return None


@contextlib.contextmanager
def scoped[T](var: contextvars.ContextVar[T], value: T) -> Generator[None]:
    token = var.set(value)
    try:
        yield
    finally:
        var.reset(token)


def map_strings(value: Any, change: Callable[[str], str], *, keys: bool = False) -> Any:
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


async def aget_active_user(identifier: str) -> AbstractBaseUser | None:
    from django.contrib.auth import get_user_model

    if not identifier:
        return None
    user_model = get_user_model()
    # AbstractBaseUser declares no USERNAME_FIELD; concrete user models do.
    field: str = user_model.USERNAME_FIELD  # ty: ignore[invalid-attribute-access, invalid-assignment]
    try:
        return await user_model._default_manager.aget(**{field: identifier, "is_active": True})
    except (user_model.DoesNotExist, user_model.MultipleObjectsReturned):
        return None
