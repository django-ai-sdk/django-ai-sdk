from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django_ai_sdk.generators.base import (
    build_kwargs,
    requires_generator,
    resolve_secret,
    resolve_setting,
)

if TYPE_CHECKING:
    from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator

# The Anthropic API has no schema parameter; structured output needs a tool call.
SCHEMA_KWARGS: dict[str, str | None] = {"AnthropicChatGenerator": None}


def anthropic_chat(**kwargs: Any) -> AnthropicChatGenerator:
    """Anthropic Messages API generator wired to Django settings."""
    with requires_generator("anthropic"):
        from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator

    return AnthropicChatGenerator(
        **build_kwargs(
            {
                "api_key": resolve_secret("ANTHROPIC_API_KEY"),
                "api_base_url": resolve_setting("ANTHROPIC_API_URL"),
            },
            kwargs,
        )
    )
