from __future__ import annotations

from typing import Any

from haystack.components.generators.chat import OpenAIChatGenerator, OpenAIResponsesChatGenerator

from django_ai_sdk.generators.base import build_kwargs, resolve_secret, resolve_setting


def openai_chat(**kwargs: Any) -> OpenAIChatGenerator:
    """OpenAI Chat Completions generator wired to Django settings."""
    return OpenAIChatGenerator(
        **build_kwargs(
            {
                "api_key": resolve_secret("OPENAI_API_KEY"),
                "api_base_url": resolve_setting("OPENAI_API_URL"),
            },
            kwargs,
        )
    )


def openai_responses_chat(**kwargs: Any) -> OpenAIResponsesChatGenerator:
    """OpenAI Responses API generator wired to Django settings."""
    return OpenAIResponsesChatGenerator(
        **build_kwargs(
            {
                "api_key": resolve_secret("OPENAI_API_KEY"),
                "api_base_url": resolve_setting("OPENAI_API_URL"),
            },
            kwargs,
        )
    )
