from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django_ai_sdk.generators.base import (
    build_kwargs,
    requires_generator,
    resolve_secret,
    resolve_setting,
)

if TYPE_CHECKING:
    from haystack_integrations.components.generators.openrouter import OpenRouterChatGenerator


# OpenRouterChatGenerator subclasses OpenAIChatGenerator, so `response_format` is
# inherited through the MRO.
SCHEMA_KWARGS: dict[str, str | None] = {}


def openrouter_chat(**kwargs: Any) -> OpenRouterChatGenerator:
    """OpenRouter chat generator wired to Django settings."""
    with requires_generator("openrouter"):
        from haystack_integrations.components.generators.openrouter import OpenRouterChatGenerator

    return OpenRouterChatGenerator(
        **build_kwargs(
            {
                "api_key": resolve_secret("OPENROUTER_API_KEY"),
                "api_base_url": resolve_setting("OPENROUTER_API_URL"),
            },
            kwargs,
        )
    )
