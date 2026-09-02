"""Per-integration configuration, read from AI_SDK_INTEGRATIONS.

One namespaced settings dict, keyed by registry name, in the shape Django uses for
every other pluggable backend (DATABASES, CACHES, STORAGES)::

    AI_SDK_INTEGRATIONS = {
        "github": {"TOKEN": env("GITHUB_MCP_TOKEN")},
        "linear": {"TOKEN": env("LINEAR_API_KEY"), "TOOLS": ["list_issues"]},
    }

INSTALLED_APPS decides which integrations exist; this dict only configures them. Keys
are upper-cased on read, so the dict reads like the rest of settings.py and a
lower-case key still resolves.

Config lives in settings, not the environment. A derived name like AI_SDK_GITHUB_TOKEN
appears nowhere in the code that reads it, and unprefixed names collide with variables
that already exist -- GitHub Actions injects its own GITHUB_TOKEN into every workflow
step. ``env("GITHUB_MCP_TOKEN")`` in settings.py names the variable outright.

Reading secrets from a vault or an ini file needs no hook here: the dict is ordinary
Python evaluated at settings-import time, so call whatever you like inside it.

Resolution never raises. This runs from IntegrationAppConfig.ready(), so an exception
would be a failed boot rather than a degraded integration -- exactly the outcome
build_mcp_config_safe() and `needs_setup` exist to prevent. A missing or malformed
entry is logged and treated as unconfigured.
"""

from __future__ import annotations

import logging
from typing import Any

from django_ai_sdk.utils import resolve_setting

logger = logging.getLogger(__name__)


def get_integration_config(integration_name: str) -> dict[str, Any]:
    """This integration's slice of AI_SDK_INTEGRATIONS, with upper-cased keys.

    Returns an empty dict when the setting or the entry is missing, or when either is
    the wrong type -- an unconfigured integration must report that it needs setup, not
    break app boot.
    """
    configured = resolve_setting("AI_SDK_INTEGRATIONS") or {}
    if not isinstance(configured, dict):
        logger.error(
            "AI_SDK_INTEGRATIONS must be a dict of {name: {KEY: value}}, got %s; "
            "treating every integration as unconfigured",
            type(configured).__name__,
        )
        return {}

    entry = configured.get(integration_name) or {}
    if not isinstance(entry, dict):
        logger.error(
            "AI_SDK_INTEGRATIONS[%r] must be a dict of {KEY: value}, got %s; treating "
            "%r as unconfigured",
            integration_name,
            type(entry).__name__,
            integration_name,
        )
        return {}

    return {str(key).upper(): value for key, value in entry.items()}


def configured_names() -> set[str]:
    """Every integration name AI_SDK_INTEGRATIONS has an entry for.

    Used to warn about config left for an integration whose app was never installed;
    see registry.get_all_integrations().
    """
    configured = resolve_setting("AI_SDK_INTEGRATIONS") or {}
    return set(configured) if isinstance(configured, dict) else set()
