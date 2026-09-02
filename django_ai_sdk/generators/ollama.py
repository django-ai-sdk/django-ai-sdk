from __future__ import annotations

from typing import Any

from haystack_integrations.components.generators.ollama import OllamaChatGenerator

from django_ai_sdk.generators.base import build_kwargs, resolve_setting


def ollama_chat(**kwargs: Any) -> OllamaChatGenerator:
    """Ollama chat generator wired to Django settings."""
    # Ollama takes reasoning as an init param, so lift it out of the generation
    # kwargs an agent declares in `llm_kwargs`.
    generation_kwargs = dict(kwargs.pop("generation_kwargs", None) or {})
    think = kwargs.pop("think", generation_kwargs.pop("think", None))
    return OllamaChatGenerator(
        **build_kwargs(
            {
                "url": resolve_setting("OLLAMA_API_URL"),
                "think": think,
                "generation_kwargs": generation_kwargs or None,
            },
            kwargs,
        )
    )
