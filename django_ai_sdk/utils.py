from __future__ import annotations

from typing import Any

from django.conf import settings


def resolve_setting(setting_name: str, default: Any = None) -> Any:
    """Read a Django setting, falling back to `default` when it is unset."""
    return getattr(settings, setting_name, default)
