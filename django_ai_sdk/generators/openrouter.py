from __future__ import annotations

from typing import Any

from haystack_integrations.components.generators.openrouter import OpenRouterChatGenerator

from django_ai_sdk.generators.base import build_kwargs, resolve_secret
from django_ai_sdk.utils import resolve_setting


def openrouter_chat(**kwargs: Any) -> OpenRouterChatGenerator:
    """OpenRouter chat generator wired to Django settings."""
    return OpenRouterChatGenerator(
        **build_kwargs(
            {
                "api_key": resolve_secret("OPENROUTER_API_KEY"),
                "api_base_url": resolve_setting("OPENROUTER_API_URL"),
            },
            kwargs,
        )
    )
