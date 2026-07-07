"""Resolves the AI_SDK_INTEGRATIONS setting into Integration instances.

Each value in AI_SDK_INTEGRATIONS can be:
  - an MCP config object (Static/Token/OAuthMCPIntegrationConfig) — resolved to an
    MCPIntegration, or
  - a dotted path to an Integration subclass or instance, or
  - an Integration instance directly.

Instances are built once per process and cached.
"""

from __future__ import annotations

from typing import Any

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.test.signals import setting_changed
from django.utils.module_loading import import_string

from django_ai_sdk.integrations.base import Integration
from django_ai_sdk.integrations.mcp.loader import MCPIntegration
from django_ai_sdk.integrations.mcp.schemas import (
    OAuthMCPIntegrationConfig,
    StaticMCPIntegrationConfig,
    TokenMCPIntegrationConfig,
)

_MCP_CONFIG_TYPES = (
    StaticMCPIntegrationConfig,
    TokenMCPIntegrationConfig,
    OAuthMCPIntegrationConfig,
)

_registry_cache: dict[str, Integration] | None = None


def _build(name: str, value: Any) -> Integration:
    if isinstance(value, Integration):
        integration = value
    elif isinstance(value, str):
        resolved = import_string(value)
        integration = resolved() if isinstance(resolved, type) else resolved
    elif isinstance(value, _MCP_CONFIG_TYPES):
        integration = MCPIntegration(name, value)
    else:
        raise ImproperlyConfigured(
            f"AI_SDK_INTEGRATIONS[{name!r}] = {value!r} is not a recognized integration. "
            "Use a Static/Token/OAuthMCPIntegrationConfig, a dotted path to an "
            "Integration subclass or instance, or an Integration instance directly."
        )

    # Fill in the name from the settings key if the integration didn't set one.
    if not getattr(integration, "name", None):
        integration.name = name
    return integration


def get_all_integrations() -> dict[str, Integration]:
    """Build every configured Integration from AI_SDK_INTEGRATIONS, once per process."""
    global _registry_cache
    if _registry_cache is not None:
        return _registry_cache

    configured: dict[str, Any] = getattr(settings, "AI_SDK_INTEGRATIONS", {})
    _registry_cache = {name: _build(name, value) for name, value in configured.items()}
    return _registry_cache


def get_integrations(names: list[str]) -> dict[str, Integration]:
    """Return the configured Integrations named in ``names``. Unknown names are skipped."""
    all_integrations = get_all_integrations()
    return {name: all_integrations[name] for name in names if name in all_integrations}


def reset_registry_cache() -> None:
    """Clear the resolved-integrations cache, forcing a rebuild on next access."""
    global _registry_cache
    _registry_cache = None


def _on_setting_changed(*, setting: str, **kwargs: Any) -> None:
    """Auto-invalidate when AI_SDK_INTEGRATIONS changes under ``override_settings``.

    Without this, a test suite using ``override_settings(AI_SDK_INTEGRATIONS=...)``
    would silently keep serving integrations built from the previous value, since
    the cache is otherwise process-lifetime.
    """
    if setting == "AI_SDK_INTEGRATIONS":
        reset_registry_cache()


setting_changed.connect(_on_setting_changed)
