from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from django.conf import settings
from haystack.utils import Secret

if TYPE_CHECKING:
    from collections.abc import Generator


def resolve_secret(setting_name: str) -> Secret | None:
    """Wrap a Django setting as a Secret, or None when it is unset."""
    value = getattr(settings, setting_name, None)
    if not value:
        return None
    return Secret.from_token(value)


def resolve_setting(setting_name: str) -> Any:
    """Read an optional Django setting."""
    return getattr(settings, setting_name, None)


def build_kwargs(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge caller kwargs over settings defaults."""
    return {key: value for key, value in {**defaults, **overrides}.items() if value is not None}


def merge_generation_kwargs(
    base: dict[str, Any] | None,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    """Shallow-merge generation kwargs, `extra` wins."""
    return {**(base or {}), **(extra or {})}


@contextmanager
def requires_generator(extra: str) -> Generator[None]:
    """Wrap an optional generator import and name the extra that provides it."""
    try:
        yield
    except ImportError as e:
        raise ImportError(f"This generator needs: pip install django-ai-sdk[{extra}]") from e
