"""Per-integration secret resolution.

An integration reads its own secrets and config through get_integration_secret().
The default resolver reads {INTEGRATION}_{KEY} from the environment, e.g.
LINEAR_TOKEN — no settings.py wiring needed for the common case. A project can
instead point AI_SDK_INTEGRATION_SECRETS at its own resolver (e.g. one backed by
an ini file) with the same (integration_name, key) -> str | None signature; that
setting names which resolver to use, it holds no secret values itself.
"""

from __future__ import annotations

import os

from django.conf import settings
from django.utils.module_loading import import_string


def env_secret(integration_name: str, key: str) -> str | None:
    """Default resolver: reads {INTEGRATION}_{KEY} from the environment."""
    return os.environ.get(f"{integration_name.upper()}_{key.upper()}")


def get_integration_secret(integration_name: str, key: str, default: str = "") -> str:
    """Resolve one secret value for integration_name, via AI_SDK_INTEGRATION_SECRETS."""
    path = getattr(settings, "AI_SDK_INTEGRATION_SECRETS", None)
    resolver = import_string(path) if path else env_secret
    value = resolver(integration_name, key)
    return value if value else default
