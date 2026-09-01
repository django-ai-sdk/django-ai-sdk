from __future__ import annotations

from django_ai_sdk.generators.anthropic import anthropic_chat
from django_ai_sdk.generators.azure import azure_openai_chat, azure_openai_responses_chat
from django_ai_sdk.generators.base import (
    build_kwargs,
    merge_generation_kwargs,
    requires_generator,
)
from django_ai_sdk.generators.huggingface import huggingface_api_chat, transformers_chat
from django_ai_sdk.generators.mistral import mistral_chat
from django_ai_sdk.generators.ollama import ollama_chat
from django_ai_sdk.generators.openai import (
    add_usage_reporting,
    openai_chat,
    openai_responses_chat,
)
from django_ai_sdk.generators.openrouter import openrouter_chat
from django_ai_sdk.generators.schema import schema_kwargs

__all__ = [
    "add_usage_reporting",
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
    "requires_generator",
    "schema_kwargs",
    "transformers_chat",
]
