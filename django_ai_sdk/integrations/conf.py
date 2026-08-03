"""Per-integration configuration resolution.

An integration reads its own secrets and config through get_integration_config().
The default resolver reads {INTEGRATION}_{KEY} from the environment; a project can
point AI_SDK_INTEGRATION_CONFIG at its own resolver (e.g. one backed by an ini file)
with the same signature.
"""

from __future__ import annotations

import os

from django.conf import settings
from django.utils.module_loading import import_string


def env_config(integration_name: str, key: str) -> str | None:
    """Default resolver: reads {INTEGRATION}_{KEY} from the environment."""
    return os.environ.get(f"{integration_name.upper()}_{key.upper()}")


def get_integration_config(integration_name: str, key: str, default: str = "") -> str:
    """Resolve one config value for integration_name, via AI_SDK_INTEGRATION_CONFIG."""
    path = getattr(settings, "AI_SDK_INTEGRATION_CONFIG", None)
    resolver = import_string(path) if path else env_config
    value = resolver(integration_name, key)
    return value if value else default
