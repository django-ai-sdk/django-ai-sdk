from __future__ import annotations

from importlib import import_module
from typing import Any

from django_ai_sdk.generators.base import build_kwargs, merge_generation_kwargs
from django_ai_sdk.generators.openai import openai_chat, openai_responses_chat
from django_ai_sdk.generators.schema import schema_kwargs

_LAZY = {
    "anthropic_chat": "django_ai_sdk.generators.anthropic",
    "azure_openai_chat": "django_ai_sdk.generators.azure",
    "azure_openai_responses_chat": "django_ai_sdk.generators.azure",
    "huggingface_api_chat": "django_ai_sdk.generators.huggingface",
    "mistral_chat": "django_ai_sdk.generators.mistral",
    "ollama_chat": "django_ai_sdk.generators.ollama",
    "openrouter_chat": "django_ai_sdk.generators.openrouter",
    "transformers_chat": "django_ai_sdk.generators.transformers",
}


def __getattr__(name: str) -> Any:
    """Import a vendor's factory on first use (PEP 562)."""
    if module := _LAZY.get(name):
        return getattr(import_module(module), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "anthropic_chat",
    "azure_openai_chat",
    "azure_openai_responses_chat",
    "build_kwargs",
    "huggingface_api_chat",
    "merge_generation_kwargs",
    "mistral_chat",
    "ollama_chat",
    "openai_chat",
    "openai_responses_chat",
    "openrouter_chat",
    "schema_kwargs",
    "transformers_chat",
]
