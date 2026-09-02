from __future__ import annotations

from typing import Any

from haystack.utils import Secret

from django_ai_sdk.utils import resolve_setting


def resolve_secret(setting_name: str) -> Secret | None:
    """Wrap a Django setting as a Secret, or None when it is unset."""
    value = resolve_setting(setting_name)
    if not value:
        return None
    return Secret.from_token(value)


def build_kwargs(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge caller kwargs over settings defaults."""
    return {key: value for key, value in {**defaults, **overrides}.items() if value is not None}


def merge_generation_kwargs(
    base: dict[str, Any] | None,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    """Shallow-merge generation kwargs, `extra` wins."""
    return {**(base or {}), **(extra or {})}
