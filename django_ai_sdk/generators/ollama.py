from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django_ai_sdk.generators.base import build_kwargs, requires_generator, resolve_setting

if TYPE_CHECKING:
    from haystack_integrations.components.generators.ollama import OllamaChatGenerator

# Ollama wants a JSON schema at construction, not a run-time kwarg:
# ollama_chat(response_format=Schema.model_json_schema()).
SCHEMA_KWARGS: dict[str, str | None] = {"OllamaChatGenerator": None}


def ollama_chat(**kwargs: Any) -> OllamaChatGenerator:
    """Ollama chat generator wired to Django settings."""
    with requires_generator("ollama"):
        from haystack_integrations.components.generators.ollama import OllamaChatGenerator

    return OllamaChatGenerator(
        **build_kwargs(
            {"url": resolve_setting("OLLAMA_API_URL")},
            kwargs,
        )
    )
