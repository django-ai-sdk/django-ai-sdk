from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django_ai_sdk.generators.base import (
    build_kwargs,
    requires_generator,
    resolve_secret,
    resolve_setting,
)

if TYPE_CHECKING:
    from haystack_integrations.components.generators.mistral import MistralChatGenerator


# MistralChatGenerator subclasses OpenAIChatGenerator, so `response_format` is
# inherited through the MRO.
SCHEMA_KWARGS: dict[str, str | None] = {}


def mistral_chat(**kwargs: Any) -> MistralChatGenerator:
    """Mistral chat generator wired to Django settings."""
    with requires_generator("mistral"):
        from haystack_integrations.components.generators.mistral import MistralChatGenerator

    return MistralChatGenerator(
        **build_kwargs(
            {
                "api_key": resolve_secret("MISTRAL_API_KEY"),
                "api_base_url": resolve_setting("MISTRAL_API_URL"),
            },
            kwargs,
        )
    )
