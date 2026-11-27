from __future__ import annotations

from typing import Any

from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator

from django_ai_sdk.generators.base import build_kwargs, resolve_secret
from django_ai_sdk.oversight.calls import watched_factory


@watched_factory
def anthropic_chat(**kwargs: Any) -> AnthropicChatGenerator:
    # AnthropicChatGenerator has no base-URL parameter.
    return AnthropicChatGenerator(
        **build_kwargs({"api_key": resolve_secret("ANTHROPIC_API_KEY")}, kwargs)
    )
