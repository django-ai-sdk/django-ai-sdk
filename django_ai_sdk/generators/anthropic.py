from __future__ import annotations

from typing import Any

from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator

from django_ai_sdk.generators.base import build_kwargs, resolve_secret


def anthropic_chat(**kwargs: Any) -> AnthropicChatGenerator:
    """Anthropic Messages API generator wired to Django settings."""
    # AnthropicChatGenerator has no base-URL parameter, so there is no
    # ANTHROPIC_API_URL setting to honour.
    return AnthropicChatGenerator(
        **build_kwargs({"api_key": resolve_secret("ANTHROPIC_API_KEY")}, kwargs)
    )
