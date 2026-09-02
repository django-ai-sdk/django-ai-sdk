from __future__ import annotations

from typing import Any

from haystack_integrations.components.generators.mistral import MistralChatGenerator

from django_ai_sdk.generators.base import (
    build_kwargs,
    resolve_secret,
    resolve_setting,
)


def mistral_chat(**kwargs: Any) -> MistralChatGenerator:
    """Mistral chat generator wired to Django settings."""
    return MistralChatGenerator(
        **build_kwargs(
            {
                "api_key": resolve_secret("MISTRAL_API_KEY"),
                "api_base_url": resolve_setting("MISTRAL_API_URL"),
            },
            kwargs,
        )
    )
