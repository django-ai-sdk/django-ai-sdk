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
